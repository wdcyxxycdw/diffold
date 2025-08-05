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
        """计算结构对齐后的RMSD (使用Kabsch算法)"""
        # 确保坐标形状匹配
        if pred_coords.shape != target_coords.shape:
            min_len = min(pred_coords.shape[1], target_coords.shape[1])
            pred_coords = pred_coords[:, :min_len]
            target_coords = target_coords[:, :min_len]
        
        # 检查坐标维度和有效性
        if pred_coords.dim() != 3 or target_coords.dim() != 3:
            logger.warning(f"坐标维度错误: pred {pred_coords.shape}, target {target_coords.shape}")
            # 回退到原始计算
            diff = pred_coords - target_coords
            squared_diff = torch.sum(diff ** 2, dim=-1)
            return torch.sqrt(torch.mean(squared_diff, dim=-1))
        
        if pred_coords.shape[-1] != 3 or target_coords.shape[-1] != 3:
            logger.warning(f"坐标不是3D: pred {pred_coords.shape}, target {target_coords.shape}")
            # 回退到原始计算
            diff = pred_coords - target_coords
            squared_diff = torch.sum(diff ** 2, dim=-1)
            return torch.sqrt(torch.mean(squared_diff, dim=-1))
        
        # 快速检查：如果坐标完全相同，直接返回0
        if torch.allclose(pred_coords, target_coords, atol=1e-6):
            return torch.zeros(pred_coords.shape[0], device=pred_coords.device)
        
        try:
            return self._compute_aligned_rmsd_kabsch(pred_coords, target_coords)
        except Exception as e:
            logger.warning(f"Kabsch对齐失败，使用未对齐RMSD: {e}")
            # 回退到原始计算
            diff = pred_coords - target_coords
            squared_diff = torch.sum(diff ** 2, dim=-1)
            return torch.sqrt(torch.mean(squared_diff, dim=-1))
    
    def _compute_aligned_rmsd_kabsch(self, 
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
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # 快速检查：如果坐标完全相同，直接返回0
        if torch.allclose(pred_coords, target_coords, atol=1e-6):
            return torch.zeros(batch_size, device=device)
        
        # 1. 质心对齐
        pred_centroid = torch.mean(pred_coords, dim=1, keepdim=True)  # [B, 1, 3]
        target_centroid = torch.mean(target_coords, dim=1, keepdim=True)  # [B, 1, 3]
        
        pred_centered = pred_coords - pred_centroid  # [B, N, 3]
        target_centered = target_coords - target_centroid  # [B, N, 3]
        
        # 2. 计算交叉协方差矩阵 H = pred_centered^T @ target_centered
        H = torch.bmm(pred_centered.transpose(-2, -1), target_centered)  # [B, 3, 3]
        
        # 3. SVD分解: H = U @ S @ V^T
        try:
            # 使用稳定的SVD分解
            U, S, Vt = torch.linalg.svd(H) if hasattr(torch.linalg, 'svd') else torch.svd(H)
        except Exception as e:
            logger.warning(f"SVD分解失败: {e}")
            # 回退到未对齐RMSD
            diff = pred_coords - target_coords
            return torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1), dim=-1))
        
        # 4. 计算旋转矩阵 R = V @ U^T
        # 注意：这里V是Vt的转置
        V = Vt.transpose(-2, -1)  # [B, 3, 3]
        R = torch.bmm(U, Vt)  # [B, 3, 3]
        
        # 5. 处理反射情况：确保det(R) = +1（右手系）
        det_R = torch.det(R)  # [B]
        
        # 对于det(R) < 0的情况，修正V的最后一列
        det_R = torch.det(R)
        neg_mask = det_R < 0
        if neg_mask.any():
            # 在 U 的最后一列乘 -1 再重算 R
            U_fix = U.clone()
            U_fix[neg_mask, :, -1] *= -1
            R[neg_mask] = torch.bmm(U_fix[neg_mask], Vt[neg_mask])
        
        # 6. 应用最优旋转：pred_aligned = pred_centered @ R
        pred_aligned = torch.bmm(pred_centered, R)  # [B, N, 3]
        
        # 7. 计算对齐后的RMSD
        diff = pred_aligned - target_centered  # [B, N, 3]
        squared_distances = torch.sum(diff ** 2, dim=-1)  # [B, N]
        rmsd = torch.sqrt(torch.mean(squared_distances, dim=-1))  # [B]
        
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
        参考Zhang Lab的RNA-align方法，使用适合RNA的归一化因子
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            target_coords: [batch_size, n_atoms, 3] 目标坐标
            
        Returns:
            tm_score: [batch_size] 每个样本的RNA TM-score
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # 确保坐标形状匹配
        if pred_coords.shape != target_coords.shape:
            min_len = min(pred_coords.shape[1], target_coords.shape[1])
            pred_coords = pred_coords[:, :min_len]
            target_coords = target_coords[:, :min_len]
            n_atoms = min_len
        
        try:
            # 1. 获取最优对齐（使用与RMSD相同的对齐方法）
            pred_centroid = torch.mean(pred_coords, dim=1, keepdim=True)
            target_centroid = torch.mean(target_coords, dim=1, keepdim=True)
            
            pred_centered = pred_coords - pred_centroid
            target_centered = target_coords - target_centroid
            
            # 计算旋转矩阵
            H = torch.bmm(pred_centered.transpose(-2, -1), target_centered)
            
            # SVD分解获得最优旋转
            U, S, Vt = torch.linalg.svd(H) if hasattr(torch.linalg, 'svd') else torch.svd(H)
            V = Vt.transpose(-2, -1)
            R = torch.bmm(U, Vt)
            
            # 处理反射
            det_R = torch.det(R)
            neg_mask = det_R < 0
            if neg_mask.any():
                U_fix = U.clone()
                U_fix[neg_mask, :, -1] *= -1
                R[neg_mask] = torch.bmm(U_fix[neg_mask], Vt[neg_mask])
            
            # 应用旋转
            pred_aligned = torch.bmm(pred_centered, R)
            
            # 2. 计算距离
            distances = torch.sqrt(torch.sum((pred_aligned - target_centered) ** 2, dim=-1))  # [B, N]
            
            # 3. RNA TM-score计算 - 使用RNA优化的d0公式
            L = n_atoms
            
            # RNA专用的d0计算（基于RNA-align方法）
            if L > 30:
                d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8 + 0.3  # 增加0.3埃的偏移
                d0 = max(d0, 0.5)  # 最小值保持为0.5
            else:
                d0 = 0.5 + 0.3 * (L / 30.0)  # 短序列的平滑过渡
            
            # TM-score = 1/L * sum(1 / (1 + (di/d0)^2))
            d0_squared = d0 ** 2
            tm_scores = torch.mean(1.0 / (1.0 + distances ** 2 / d0_squared), dim=-1)  # [B]
            
            return torch.clamp(tm_scores, min=0.0, max=1.0)
            
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
        
        metrics = {
            'avg_loss': self.total_loss / self.total_samples,
            'batch_count': self.batch_count,
            'total_samples': self.total_samples,
            'structure_type': 'RNA'  # 专门为RNA设计
        }
        
        # 添加损失分解
        for component, total_value in self.loss_components.items():
            metrics[f'avg_{component}'] = total_value / self.total_samples
        
        # 添加RNA结构指标
        if self.rmsd_values:
            import numpy as np
            metrics['avg_rmsd'] = np.mean(self.rmsd_values)
            metrics['median_rmsd'] = np.median(self.rmsd_values)
            metrics['std_rmsd'] = np.std(self.rmsd_values)
        
        # 添加RNA TM-score指标
        if self.tm_scores:
            import numpy as np
            metrics['avg_tm_score'] = np.mean(self.tm_scores)
            metrics['median_tm_score'] = np.median(self.tm_scores)
            metrics['std_tm_score'] = np.std(self.tm_scores)
            
            # RNA特有的阈值评估
            # RNA family similarity threshold (≥0.45 for same Rfam family)
            good_predictions = np.sum(np.array(self.tm_scores) >= 0.45)
            metrics['tm_score_good_ratio'] = good_predictions / len(self.tm_scores)
            
            # Excellent predictions (≥0.6)
            excellent_predictions = np.sum(np.array(self.tm_scores) >= 0.6)
            metrics['tm_score_excellent_ratio'] = excellent_predictions / len(self.tm_scores)
        
        # 添加RNA lDDT指标
        if self.lddt_scores:
            import numpy as np
            metrics['avg_lddt'] = np.mean(self.lddt_scores)
            metrics['median_lddt'] = np.median(self.lddt_scores)
            metrics['std_lddt'] = np.std(self.lddt_scores)
            
            # lDDT质量分级
            lddt_array = np.array(self.lddt_scores)
            metrics['lddt_high_quality_ratio'] = np.sum(lddt_array >= 70.0) / len(self.lddt_scores)  # 高质量
            metrics['lddt_good_quality_ratio'] = np.sum(lddt_array >= 50.0) / len(self.lddt_scores)   # 良好质量
        
        # 添加RNA clash score指标
        if self.clash_scores:
            import numpy as np
            metrics['avg_clash_score'] = np.mean(self.clash_scores)
            metrics['median_clash_score'] = np.median(self.clash_scores)
            metrics['std_clash_score'] = np.std(self.clash_scores)
            
            # Clash score质量评估 (值越低越好)
            clash_array = np.array(self.clash_scores)
            metrics['clash_low_ratio'] = np.sum(clash_array <= 5.0) / len(self.clash_scores)  # 低冲突比例
        
        # 添加置信度指标
        if self.confidence_scores:
            import numpy as np
            metrics['avg_confidence'] = np.mean(self.confidence_scores)
            metrics['median_confidence'] = np.median(self.confidence_scores)
        
        return metrics


########################################
#              UNIT TESTS              #
########################################

def _random_rotation(batch: int, device: torch.device):
    """Generate a batch of random 3×3 rotation matrices (det=+1)."""
    rand = torch.randn(batch, 3, 3, device=device)
    U, _, Vh = torch.linalg.svd(rand)
    V = Vh.transpose(1, 2)
    R = torch.bmm(V, U.transpose(1, 2))
    det = torch.det(R)
    mask = det < 0.0
    if mask.any():
        V[mask, :, -1] *= -1
        R[mask] = torch.bmm(V[mask], U.transpose(1, 2)[mask])
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_kabsch_alignment()
    test_metrics_pipeline()
    print("All tests passed ✔")