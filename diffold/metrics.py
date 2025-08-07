import torch
from typing import Dict, List
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class RNAEvaluationMetrics:
    """RNA结构评估指标计算器 - 专为RNA结构预测设计"""
    
    def __init__(self):
        """初始化RNA评估指标"""
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.batch_count = 0
        
        # 损失分解
        self.loss_components = defaultdict(float)
        
        # RNA核心结构评估指标
        self.rmsd_values = []
        self.tm_scores = []  # RNA TM-score
        self.lddt_scores = []  # RNA lDDT
        self.clash_scores = []  # RNA clash score
        
        # 置信度指标
        self.confidence_scores = []
    
    def update(self, 
              loss: float, 
              batch_size: int,
              loss_breakdown: Dict[str, float] = None,
              predicted_coords: torch.Tensor = None,
              target_coords: torch.Tensor = None,
              confidence_scores: torch.Tensor = None):
        """更新指标"""
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.batch_count += 1
        
        # 更新损失分解
        if loss_breakdown:
            for component, value in loss_breakdown.items():
                self.loss_components[component] += value * batch_size
        
        # 更新RNA结构指标
        if predicted_coords is not None and target_coords is not None:
            self._update_rna_structure_metrics(predicted_coords, target_coords)
        
        # 更新置信度指标
        if confidence_scores is not None:
            # 检查 confidence_scores 的类型
            if hasattr(confidence_scores, 'plddt') and confidence_scores.plddt is not None:
                # ConfidenceHeadLogits 对象，提取 plddt 分数
                plddt_scores = confidence_scores.plddt
                if torch.is_tensor(plddt_scores):
                    self.confidence_scores.extend(plddt_scores.flatten().tolist())
            elif torch.is_tensor(confidence_scores):
                # 普通张量
                self.confidence_scores.extend(confidence_scores.flatten().tolist())
    
    def _update_rna_structure_metrics(self, 
                                    predicted_coords: torch.Tensor, 
                                    target_coords: torch.Tensor):
        """更新RNA结构评估指标"""
        try:
            # 计算RMSD
            rmsd = self._compute_rmsd(predicted_coords, target_coords)
            self.rmsd_values.extend(rmsd.tolist())
            
            # 计算RNA TM-score
            tm_scores = self._compute_rna_tm_score(predicted_coords, target_coords)
            self.tm_scores.extend(tm_scores.tolist())
            
            # 计算RNA lDDT
            lddt_scores = self._compute_rna_lddt(predicted_coords, target_coords)
            self.lddt_scores.extend(lddt_scores.tolist())
            
            # 计算RNA clash score
            clash_scores = self._compute_rna_clash_score(predicted_coords)
            self.clash_scores.extend(clash_scores.tolist())
            
        except Exception as e:
            logger.warning(f"RNA结构指标计算失败: {e}")
    
    def _compute_rmsd(self, 
                                   pred_coords: torch.Tensor, 
                                   target_coords: torch.Tensor) -> torch.Tensor:
        """使用标准Kabsch算法计算对齐后的RMSD
        
        标准Kabsch算法实现，参考经典论文和可靠实现
        W. Kabsch (1976) "A solution for the best rotation to relate two sets of vectors"
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标 (要被对齐的点集)
            target_coords: [batch_size, n_atoms, 3] 目标坐标 (参考点集)
            
        Returns:
            rmsd: [batch_size] 每个样本的对齐RMSD
        """
        batch_size=pred_coords.shape[0]

        # 1. 质心对齐
        pred_centroid = torch.mean(pred_coords, dim=1, keepdim=True)  # [B, 1, 3]
        target_centroid = torch.mean(target_coords, dim=1, keepdim=True)  # [B, 1, 3]
        
        pred_centered = pred_coords - pred_centroid  # [B, N, 3]
        target_centered = target_coords - target_centroid  # [B, N, 3]
        
        # 2. 计算交叉协方差矩阵 H = pred_centered^T @ target_centered
        H = torch.bmm(pred_centered.transpose(-2, -1), target_centered)  # [B, 3, 3]
        
        # 3. SVD分解: H = U @ S @ V^T
        U, _, Vt = torch.linalg.svd(H, full_matrices=False)
        
        # 4. 计算旋转矩阵 R = V @ U^T
        # 注意：这里V是Vt的转置
        Ut = U.transpose(-2, -1)
        V = Vt.transpose(-2, -1)  # [B, 3, 3]
        
        # 5. 处理反射情况：确保det(R) = +1（右手系）
        # 对于det(R) < 0的情况，修正V的最后一列
        det_r = torch.det(torch.bmm(V, Ut))
        d = torch.where(det_r < 0, -torch.ones_like(det_r), torch.ones_like(det_r))
        D = torch.eye(3, device=H.device).repeat(H.size(0), 1, 1)
        D[:, 2, 2] = d

        R=torch.bmm(torch.bmm(U, D), Vt)
        
        # 6. 应用最优旋转：pred_aligned = pred_centered @ R
        pred_aligned = torch.bmm(pred_centered, R)  # [B, N, 3]
        
        # 7. 计算对齐后的RMSD
        diff = pred_aligned - target_centered  # [B, N, 3]
        rmsd = torch.sqrt((diff ** 2).sum(-1).mean(-1)) # [B]
        
        # 8. 验证结果的合理性
        # 计算未对齐RMSD作为上界
        unaligned_diff = pred_coords - target_coords
        unaligned_rmsd = torch.sqrt(torch.mean(torch.sum(unaligned_diff ** 2, dim=-1), dim=-1))
        
        # 正常情况下，对齐后的RMSD应该小于等于未对齐的RMSD
        # 如果不是，说明算法有问题，但我们仍然返回结果并记录警告
        worse_alignment = rmsd > unaligned_rmsd * 1.01  # 允许1%的数值误差
        if worse_alignment.any():
            logger.warning(f"警告: {worse_alignment.sum()}/{batch_size} 个样本的对齐RMSD大于未对齐RMSD")
        
        return torch.clamp(rmsd, min=0.0)
    
    def _compute_rna_tm_score(self, 
                         pred_coords: torch.Tensor, 
                         target_coords: torch.Tensor) -> torch.Tensor:
        """计算RNA TM-score (Template Modeling score)
        
        RNA TM-score是衡量两个RNA结构相似性的指标，范围0-1，值越高越好
        使用与RMSD相同的Kabsch对齐算法，确保一致性
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            target_coords: [batch_size, n_atoms, 3] 目标坐标
            
        Returns:
            tm_score: [batch_size] 每个样本的RNA TM-score
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device

        try:
            # 1. 使用与RMSD相同的Kabsch对齐算法
            # 质心对齐
            pred_centroid = torch.mean(pred_coords, dim=1, keepdim=True)  # [B, 1, 3]
            target_centroid = torch.mean(target_coords, dim=1, keepdim=True)  # [B, 1, 3]
            
            pred_centered = pred_coords - pred_centroid  # [B, N, 3]
            target_centered = target_coords - target_centroid  # [B, N, 3]
            
            # 计算交叉协方差矩阵 H = pred_centered^T @ target_centered
            H = torch.bmm(pred_centered.transpose(-2, -1), target_centered)  # [B, 3, 3]
            
            # SVD分解: H = U @ S @ V^T
            U, _, Vt = torch.linalg.svd(H, full_matrices=False)
            
            # 计算旋转矩阵 R = V @ U^T
            Ut = U.transpose(-2, -1)
            V = Vt.transpose(-2, -1)  # [B, 3, 3]
            
            # 处理反射情况：确保det(R) = +1（右手系）
            det_r = torch.det(torch.bmm(V, Ut))
            d = torch.where(det_r < 0, -torch.ones_like(det_r), torch.ones_like(det_r))
            D = torch.eye(3, device=H.device).repeat(H.size(0), 1, 1)
            D[:, 2, 2] = d
            
            R = torch.bmm(torch.bmm(U, D), Vt)
            
            # 应用最优旋转：pred_aligned = pred_centered @ R
            pred_aligned = torch.bmm(pred_centered, R)  # [B, N, 3]
            
            # 2. 计算对齐后的距离
            distances = torch.sqrt(torch.sum((pred_aligned - target_centered) ** 2, dim=-1))  # [B, N]
            
            # 3. RNA TM-score计算 - 使用更准确的d0公式
            L = n_atoms
            
            d0 = 0.6 * (L - 0.5)**0.5 - 2.5
                        
            # TM-score = 1/L * sum(1 / (1 + (di/d0)^2))
            tm_scores = torch.mean(1.0 / (1.0 + distances ** 2 / d0 ** 2), dim=-1)  # [B]
            
            # 验证结果的合理性
            # TM-score应该在0-1范围内，对于相同结构应该接近1
            if torch.any(tm_scores > 1.0 + 1e-6):
                logger.warning(f"警告: 检测到TM-score > 1: {tm_scores[tm_scores > 1.0]}")
            
            return tm_scores
            
        except Exception as e:
            logger.warning(f"RNA TM-score计算失败: {e}")
            # 返回默认值
            return torch.zeros(batch_size, device=device)
    
    def _compute_rna_lddt(self, 
                     pred_coords: torch.Tensor, 
                     target_coords: torch.Tensor,
                     cutoff_distances: List[float] = None,
                     inclusion_radius: float = None) -> torch.Tensor:
        """计算RNA lDDT (local Distance Difference Test)
        
        RNA lDDT是基于距离差异的本地结构质量指标，范围0-100，值越高越好
        使用针对RNA调整的inclusion_radius和距离阈值
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            target_coords: [batch_size, n_atoms, 3] 目标坐标
            cutoff_distances: 距离差异的阈值列表（默认RNA优化值）
            inclusion_radius: 考虑的原子对距离半径（默认RNA优化值）
            
        Returns:
            lddt_score: [batch_size] 每个样本的RNA lDDT分数
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # RNA专用的默认参数
        if cutoff_distances is None:
            cutoff_distances = [1.0, 2.0, 4.0, 8.0]  # RNA使用更大的阈值
        
        if inclusion_radius is None:
            inclusion_radius = 20.0  # RNA使用更大的inclusion radius
        
        # 确保坐标形状匹配
        if pred_coords.shape != target_coords.shape:
            min_len = min(pred_coords.shape[1], target_coords.shape[1])
            pred_coords = pred_coords[:, :min_len]
            target_coords = target_coords[:, :min_len]
            n_atoms = min_len
        
        try:
            # 1. 计算所有原子对之间的距离
            # pred_distances: [B, N, N]
            pred_diff = pred_coords.unsqueeze(2) - pred_coords.unsqueeze(1)  # [B, N, N, 3]
            pred_distances = torch.sqrt(torch.sum(pred_diff ** 2, dim=-1))  # [B, N, N]
            
            target_diff = target_coords.unsqueeze(2) - target_coords.unsqueeze(1)  # [B, N, N, 3]
            target_distances = torch.sqrt(torch.sum(target_diff ** 2, dim=-1))  # [B, N, N]
            
            # 2. 创建mask：只考虑inclusion_radius范围内的原子对
            inclusion_mask = target_distances <= inclusion_radius  # [B, N, N]
            
            # 排除对角线（自己与自己的距离）
            diag_mask = torch.eye(n_atoms, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
            inclusion_mask = inclusion_mask & (~diag_mask)
            
            # 3. 计算距离差异
            distance_diff = torch.abs(pred_distances - target_distances)  # [B, N, N]
            
            # 4. 对每个阈值计算保存的接触数
            lddt_scores = []
            for cutoff in cutoff_distances:
                preserved = (distance_diff <= cutoff) & inclusion_mask  # [B, N, N]
                preserved_count = torch.sum(preserved.float(), dim=(1, 2))  # [B]
                total_count = torch.sum(inclusion_mask.float(), dim=(1, 2))  # [B]
                
                # 避免除零
                score = torch.where(total_count > 0, 
                                  preserved_count / total_count, 
                                  torch.zeros_like(preserved_count))
                lddt_scores.append(score)
            
            # 5. lDDT是所有阈值的平均值，乘以100
            lddt_final = torch.stack(lddt_scores, dim=0).mean(dim=0) * 100.0  # [B]
            
            return torch.clamp(lddt_final, min=0.0, max=100.0)
            
        except Exception as e:
            logger.warning(f"RNA lDDT计算失败: {e}")
            return torch.zeros(batch_size, device=device)
    
    def _compute_rna_clash_score(self, 
                           pred_coords: torch.Tensor,
                           clash_threshold: float = None,
                           vdw_radii: Dict[str, float] = None) -> torch.Tensor:
        """计算RNA clash score（原子冲突分数）
        
        检测RNA结构中过于接近的原子对，值越低越好
        使用针对RNA调整的冲突阈值
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            clash_threshold: 冲突距离阈值（埃），默认RNA优化值
            vdw_radii: 原子的范德华半径字典（暂未使用）
            
        Returns:
            clash_score: [batch_size] 每个样本的冲突分数
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # RNA专用的冲突阈值
        if clash_threshold is None:
            clash_threshold = 2.5  # RNA使用更大的阈值
        
        try:
            # 计算所有原子对之间的距离
            coord_diff = pred_coords.unsqueeze(2) - pred_coords.unsqueeze(1)  # [B, N, N, 3]
            distances = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))  # [B, N, N]
            
            # 排除对角线（自己与自己）
            mask = torch.eye(n_atoms, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
            distances = distances.masked_fill(mask, float('inf'))
            
            # 检测冲突
            clashes = distances < clash_threshold  # [B, N, N]
            
            # 计算每个样本的冲突数量（除以2因为每个冲突被计算了两次）
            clash_count = torch.sum(clashes.float(), dim=(1, 2)) / 2.0  # [B]
            
            # 归一化：除以可能的原子对数量
            total_pairs = n_atoms * (n_atoms - 1) / 2.0
            clash_score = clash_count / total_pairs * 100.0  # 转换为百分比
            
            return clash_score
            
        except Exception as e:
            logger.warning(f"RNA Clash score计算失败: {e}")
            return torch.zeros(batch_size, device=device)
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算最终RNA评估指标"""
        if self.total_samples == 0:
            return {}

        import numpy as np

        metrics: Dict[str, float] = {
            "avg_loss": self.total_loss / self.total_samples,
            "batch_count": self.batch_count,
            "total_samples": self.total_samples,
            "structure_type": "RNA",
        }

        # loss 分解
        for k, v in self.loss_components.items():
            metrics[f"avg_{k}"] = v / self.total_samples

        # 分布式同步指标
        if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
            metrics = self._sync_metrics_across_gpus(metrics)

        if self.rmsd_values:
            try:
                rmsd_arr = np.asarray(self.rmsd_values, dtype=np.float32)  # 指定数据类型以节省内存
                metrics.update(
                    avg_rmsd=float(rmsd_arr.mean()),
                    median_rmsd=float(np.median(rmsd_arr)),
                    std_rmsd=float(rmsd_arr.std()),
                )
                # 清理临时数组
                del rmsd_arr
            except Exception as e:
                logger.warning(f"RMSD指标计算失败: {e}")

        if self.tm_scores:
            try:
                tm_arr = np.asarray(self.tm_scores, dtype=np.float32)
                metrics.update(
                    avg_tm_score=float(tm_arr.mean()),
                    median_tm_score=float(np.median(tm_arr)),
                    std_tm_score=float(tm_arr.std()),
                    tm_score_good_ratio=float((tm_arr >= 0.45).mean()),
                    tm_score_excellent_ratio=float((tm_arr >= 0.60).mean()),
                )
                del tm_arr
            except Exception as e:
                logger.warning(f"TM-score指标计算失败: {e}")

        if self.lddt_scores:
            try:
                lddt_arr = np.asarray(self.lddt_scores, dtype=np.float32)
                metrics.update(
                    avg_lddt=float(lddt_arr.mean()),
                    median_lddt=float(np.median(lddt_arr)),
                    std_lddt=float(lddt_arr.std()),
                    lddt_high_quality_ratio=float((lddt_arr >= 70).mean()),
                    lddt_good_quality_ratio=float((lddt_arr >= 50).mean()),
                )
                del lddt_arr
            except Exception as e:
                logger.warning(f"lDDT指标计算失败: {e}")

        if self.clash_scores:
            try:
                clash_arr = np.asarray(self.clash_scores, dtype=np.float32)
                metrics.update(
                    avg_clash_score=float(clash_arr.mean()),
                    median_clash_score=float(np.median(clash_arr)),
                    std_clash_score=float(clash_arr.std()),
                    clash_low_ratio=float((clash_arr <= 5).mean()),
                )
                del clash_arr
            except Exception as e:
                logger.warning(f"Clash score指标计算失败: {e}")

        if self.confidence_scores:
            try:
                conf_arr = np.asarray(self.confidence_scores, dtype=np.float32)
                metrics.update(
                    avg_confidence=float(conf_arr.mean()),
                    median_confidence=float(np.median(conf_arr)),
                )
                del conf_arr
            except Exception as e:
                logger.warning(f"置信度指标计算失败: {e}")
        return metrics

    def _sync_metrics_across_gpus(self, local_metrics: Dict[str, float]) -> Dict[str, float]:
        """在分布式训练中同步指标"""
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        if world_size <= 1:
            return local_metrics
        
        # 获取当前设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 收集所有GPU的指标列表
        all_metrics = [None for _ in range(world_size)]
        dist.all_gather_object(all_metrics, local_metrics, group=None)
        
        # 合并所有GPU的指标
        global_metrics = {}
        
        # 对于数值型指标，计算全局平均
        numeric_keys = ['avg_loss', 'total_samples', 'batch_count']
        for key in numeric_keys:
            if key in local_metrics:
                values = [metrics.get(key, 0.0) for metrics in all_metrics if metrics is not None]
                global_metrics[key] = sum(values) / len(values)
        
        # 对于列表型指标，合并所有GPU的数据
        if self.rmsd_values:
            all_rmsd = []
            for metrics in all_metrics:
                if metrics and 'rmsd_values' in metrics:
                    all_rmsd.extend(metrics['rmsd_values'])
            if all_rmsd:
                import numpy as np
                rmsd_arr = np.asarray(all_rmsd)
                global_metrics.update(
                    avg_rmsd=float(rmsd_arr.mean()),
                    median_rmsd=float(np.median(rmsd_arr)),
                    std_rmsd=float(rmsd_arr.std()),
                )
        
        # 类似地处理其他指标...
        # 这里可以扩展处理tm_scores, lddt_scores等
        
        return global_metrics


########################################
#              UNIT TESTS              #
########################################

def _random_rotation(batch: int, device: torch.device):
    """Return a batch of proper (det = +1) rotation matrices, shape [B, 3, 3]."""
    A = torch.randn(batch, 3, 3, device=device)          # random Gaussian
    U, _, Vt = torch.linalg.svd(A, full_matrices=False)  # A = U Σ Vᵀ
    V        = Vt.transpose(1, 2)
    Ut       = U.transpose(1, 2)

    # 最优旋转：R = V D Uᵀ，D 用来消除可能的反射
    dets = torch.det(torch.bmm(V, Ut))                   # ±1 for each sample
    D    = torch.eye(3, device=device).repeat(batch, 1, 1)
    D[:, 2, 2] = torch.where(dets < 0, -1.0, 1.0)

    R = torch.bmm(torch.bmm(V, D), Ut)                   # 保证 det(R)=+1
    return R


def test_kabsch_alignment():
    B, N = 4, 200
    device = torch.device("cpu")
    coords = torch.randn(B, N, 3, device=device)
    R = _random_rotation(B, device)
    t = torch.randn(B, 1, 3, device=device)  # random translation
    coords_transformed = torch.bmm(coords, R) + t

    metric = RNAEvaluationMetrics()
    rmsd = metric._compute_rmsd(coords_transformed, coords)
    assert torch.max(rmsd) < 1e-4, f"Alignment failed, RMSD max = {rmsd.max()}"


def test_metrics_pipeline():
    B, N = 2, 50
    coords = torch.randn(B, N, 3)
    metric = RNAEvaluationMetrics()
    metric.update(
        loss=0.5,
        batch_size=B,
        predicted_coords=coords,
        target_coords=coords,
    )
    results = metric.compute_metrics()
    assert results["avg_rmsd"] < 1e-6
    assert results["avg_tm_score"] > 0.99
    assert results["avg_lddt"] > 99.0


def test_tm_score_calculation():
    """测试TM-score计算的正确性"""
    B, N = 1, 50
    device = torch.device("cpu")
    
    # 创建相同的坐标（应该得到TM-score ≈ 1）
    coords = torch.randn(B, N, 3, device=device)
    metric = RNAEvaluationMetrics()
    
    # 测试相同结构的TM-score
    tm_scores = metric._compute_rna_tm_score(coords, coords)
    print(f"相同结构TM-score: {tm_scores}")
    assert torch.allclose(tm_scores, torch.ones_like(tm_scores), atol=1e-6), f"相同结构TM-score应该为1，得到: {tm_scores}"
    
    # 测试随机旋转后的TM-score
    R = _random_rotation(B, device)
    t = torch.randn(B, 1, 3, device=device)
    coords_transformed = torch.bmm(coords, R) + t
    
    tm_scores_transformed = metric._compute_rna_tm_score(coords_transformed, coords)
    print(f"旋转后TM-score: {tm_scores_transformed}")
    assert torch.all(tm_scores_transformed > 0.9), f"旋转后TM-score应该仍然很高，得到: {tm_scores_transformed}"
    
    # 测试完全不同的结构（应该得到低TM-score）
    random_coords = torch.randn_like(coords)
    tm_scores_random = metric._compute_rna_tm_score(random_coords, coords)
    print(f"随机结构TM-score: {tm_scores_random}")
    assert torch.all(tm_scores_random < 0.8), f"不同结构TM-score应该较低，得到: {tm_scores_random}"
    
    print("✅ TM-score计算测试通过")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_kabsch_alignment()
    test_tm_score_calculation()
    test_metrics_pipeline()
    print("All tests passed ✔")