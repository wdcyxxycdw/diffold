from torch import nn
import torch
import os
import logging
from typing import NamedTuple
import torch.nn.functional as F

from alphafold3_pytorch import(
    InputFeatureEmbedder,
    ElucidatedAtomDiffusion,
    DiffusionModule,
    RelativePositionEncoding,
    ConfidenceHead,
    DistogramHead,
    ComputeAlignmentError,
)
from alphafold3_pytorch.alphafold3 import(
    to_pairwise_mask,
    masked_average,
    batch_repeat_interleave,
    distance_to_dgram,
    IS_RNA_INDEX,
    IS_DNA_INDEX,
)
from alphafold3_pytorch.inputs import NUM_MOLECULE_IDS
from .input_processor import process_alphafold3_input, process_multiple_alphafold3_inputs, align_input
from .mask_validator import MaskValidator

from rhofold.rhofold import RhoFold

# 设置日志
logger = logging.getLogger(__name__)

# 损失分解类，仿照AlphaFold3的LossBreakdown
class DiffoldLossBreakdown(NamedTuple):
    total_loss: torch.Tensor
    total_diffusion: torch.Tensor
    distogram: torch.Tensor
    pae: torch.Tensor
    pde: torch.Tensor
    plddt: torch.Tensor
    resolved: torch.Tensor
    confidence: torch.Tensor
    diffusion_mse: torch.Tensor
    diffusion_bond: torch.Tensor
    diffusion_smooth_lddt: torch.Tensor

class Diffold(nn.Module):
    def __init__(self, config, rhofold_checkpoint_path=None):
        super().__init__()

        self.config = config

        dim_atom_inputs = 3
        atoms_per_window = 27
        dim_atom = 128
        dim_atompair_inputs = 5
        dim_atompair = 16
        dim_input_embedder_token = 384
        dim_single = 384
        dim_pairwise = 128
        dim_token = 768
        sigma_data = 16
        num_molecule_types: int = NUM_MOLECULE_IDS

        dim_additional_token_feats = 33
        dim_single_inputs = dim_input_embedder_token + dim_additional_token_feats
        lddt_mask_nucleic_acid_cutoff = 30.0
        lddt_mask_other_cutoff = 15.0

        # 置信度相关参数
        self.num_plddt_bins = 50
        self.num_pae_bins = 64
        self.num_pde_bins = 64
        self.ignore_index = -1
        
        # 距离和PAE/PDE的bins（保持为tensor格式）
        self.register_buffer('distance_bins', torch.linspace(2, 22, 64).float())
        self.register_buffer('pae_bins', torch.linspace(0.5, 32, 64).float())
        self.register_buffer('pde_bins', torch.linspace(0.5, 32, 64).float())
        
        # LDDT计算相关参数
        self.lddt_mask_nucleic_acid_cutoff = lddt_mask_nucleic_acid_cutoff
        self.lddt_mask_other_cutoff = lddt_mask_other_cutoff
        
        # 损失权重
        self.loss_confidence_weight = 1e-4
        self.loss_pae_weight = 0
        self.loss_distogram_weight = 0
        self.loss_diffusion_weight = 2.0

        # 使用 RhoFold 自带的配置
        from rhofold.config import rhofold_config
        self.rhofold = RhoFold(rhofold_config)
        
        # 加载RhoFold预训练权重
        if rhofold_checkpoint_path is None:
            rhofold_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained', 'model_20221010_params.pt')
        
        if os.path.exists(rhofold_checkpoint_path):
            logger.info(f"正在加载RhoFold预训练权重: {rhofold_checkpoint_path}")
            checkpoint = torch.load(rhofold_checkpoint_path, map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                self.rhofold.load_state_dict(checkpoint['model'])
            else:
                self.rhofold.load_state_dict(checkpoint)
            logger.info("✓ RhoFold预训练权重加载成功")
        else:
            logger.warning(f"⚠ 警告: 未找到预训练权重文件: {rhofold_checkpoint_path}")
        
        # 🔥 根据配置决定是否冻结RhoFold参数
        # 如果配置中指定了微调模式，则不预先冻结RhoFold参数
        freeze_rhofold = getattr(config, 'freeze_rhofold', True)
        if freeze_rhofold:
            # 固定RhoFold模型参数，不参与训练
            for param in self.rhofold.parameters():
                param.requires_grad = False
            logger.info("🔒 RhoFold参数已冻结（默认行为）")
        else:
            logger.info("🔓 RhoFold参数保持可训练状态（微调模式）")
        
        # 设置为评估模式
        self.rhofold.eval()

        self.relative_position_encoding = RelativePositionEncoding(
            dim_out = dim_pairwise,
            r_max = 32,
            s_max = 2,
        )

        self.input_embedder = InputFeatureEmbedder(
            num_molecule_types = num_molecule_types,
            dim_atom_inputs = dim_atom_inputs,
            dim_atompair_inputs = dim_atompair_inputs,
            atoms_per_window = atoms_per_window,
            dim_atom = dim_atom,
            dim_atompair = dim_atompair,
            dim_token = dim_input_embedder_token,
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            dim_additional_token_feats = dim_additional_token_feats,
            atom_transformer_blocks = 3,
            atom_transformer_heads = 4,
            atom_transformer_kwargs = dict()
        )

        self.diffusion = DiffusionModule(
            dim_pairwise_trunk=dim_pairwise,
            dim_pairwise_rel_pos_feats=dim_pairwise,
            atoms_per_window=atoms_per_window,
            dim_pairwise=dim_pairwise,
            sigma_data=sigma_data,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            dim_token=dim_token,
            dim_single=dim_single + dim_single_inputs,
            checkpoint=False,
            single_cond_kwargs=dict(
                num_transitions=2,
                transition_expansion_factor=2,
            ),
            pairwise_cond_kwargs=dict(
                num_transitions=2
            ),
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            token_transformer_depth=16,
            token_transformer_heads=16,
            atom_decoder_depth=3,
            atom_decoder_heads=4,
        )

        self.edm = ElucidatedAtomDiffusion(
            self.diffusion,
            sigma_data=sigma_data,
            smooth_lddt_loss_kwargs=dict(
                nucleic_acid_cutoff=lddt_mask_nucleic_acid_cutoff,
                other_cutoff=lddt_mask_other_cutoff,
            ),
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=80,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
        )
        
        # 添加训练相关的权重参数
        self.nucleotide_loss_weight = 1.0
        self.ligand_loss_weight = 1.0
        
        # 添加维度适配层，将RhoFold的输出调整到diffusion模块期望的维度
        # 使用带有Layer Normalization的适配器以提高稳定性
        self.single_dim_adapter = nn.Sequential(
            nn.Linear(256, 384),
            nn.LayerNorm(384),
            nn.ReLU(inplace=True)
        )
        
        # 添加置信度头部，仿照AlphaFold3
        # ConfidenceHead需要列表格式的bins
        distance_bins_list = torch.linspace(2, 22, 64).float().tolist()
        self.confidence_head = ConfidenceHead(
            dim_single_inputs=dim_single_inputs,
            dim_atom=dim_atom,
            atompair_dist_bins=distance_bins_list,
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            num_plddt_bins=self.num_plddt_bins,
            num_pde_bins=self.num_pde_bins,
            num_pae_bins=self.num_pae_bins,
            pairformer_depth=4,
            pairformer_kwargs=dict(),
            checkpoint=False
        )
        
        # 添加 Distogram 头部
        self.distogram_head = DistogramHead(
            dim_pairwise=dim_pairwise,
            num_dist_bins=len(distance_bins_list),
            dim_atom=dim_atom,
            atom_resolution=False,  # 在 token 级别预测距离
            checkpoint=False
        )
        
        # 添加对齐误差计算模块
        self.compute_alignment_error = ComputeAlignmentError()
        
        # LDDT阈值常量
        self.lddt_threshold_values = [0.5, 1.0, 2.0, 4.0]
        
        # 添加mask验证器
        self.mask_validator = MaskValidator(
            enable_warnings=True,
            enable_logging=True,
            strict_mode=True  # 在生产环境中可以设为True
        )
        self.mask_validator_on = False
        
    def _compute_confidence_loss(self, single_fea, single_inputs, pair_fea, denoised_atom_pos, atom_feats,
                                atom_pos_ground_truth, molecule_atom_indices, molecule_atom_lens,
                                is_molecule_types, additional_molecule_feats, mask, atom_mask):
        """计算置信度损失，仿照AlphaFold3的方式"""
        device = single_fea.device
        batch_size = single_fea.shape[0]
        atom_seq_len = atom_feats.shape[1]
        
        # 计算置信度头部的输出
        ch_logits = self.confidence_head(
            single_repr=single_fea.detach(),
            single_inputs_repr=single_inputs.detach(),
            pairwise_repr=pair_fea.detach(),
            pred_atom_pos=denoised_atom_pos.detach(),
            atom_feats=atom_feats.detach(),
            molecule_atom_indices=molecule_atom_indices,
            molecule_atom_lens=molecule_atom_lens,
            mask=mask,
            return_pae_logits=True
        )
        
        # 计算distogram头部的输出
        distogram_logits = self.distogram_head(
            pairwise_repr=pair_fea.detach(),
            molecule_atom_lens=molecule_atom_lens,
            atom_feats=atom_feats.detach()
        )
        
        # 计算PAE标签
        pae_labels = self._compute_pae_labels(
            denoised_atom_pos, atom_pos_ground_truth, molecule_atom_indices, mask
        )
        
        # 计算PDE标签
        pde_labels = self._compute_pde_labels(
            denoised_atom_pos, atom_pos_ground_truth, molecule_atom_indices, mask
        )
        
        # 计算pLDDT标签
        plddt_labels = self._compute_plddt_labels(
            denoised_atom_pos, atom_pos_ground_truth, is_molecule_types, molecule_atom_lens, atom_mask
        )
        
        # 计算distogram标签
        distogram_labels = self._compute_distogram_labels(
            denoised_atom_pos, atom_pos_ground_truth, molecule_atom_indices, mask
        )
        
        # 计算交叉熵损失
        ignore_index = self.ignore_index
        label_pairwise_mask = to_pairwise_mask(mask)
        
        # PAE损失
        if ch_logits.pae is not None:
            # 正确处理PAE logits的形状：[batch, bins, seq, seq] -> [batch, seq, seq, bins]
            pae_logits_reordered = ch_logits.pae.permute(0, 2, 3, 1)  # [batch, seq, seq, bins]
            pae_logits_flat = pae_logits_reordered.reshape(-1, pae_logits_reordered.shape[-1])  # [batch*seq*seq, bins]
            pae_labels_flat = pae_labels.reshape(-1)  # [batch*seq*seq]
            
            # 处理标签中的越界值
            num_pae_bins = pae_logits_flat.shape[-1]
            pae_labels_flat = torch.clamp(pae_labels_flat, min=0, max=num_pae_bins-1)
            
            pae_loss = F.cross_entropy(
                pae_logits_flat,
                pae_labels_flat,
                ignore_index=ignore_index,
                reduction='mean'
            )
        else:
            pae_loss = torch.tensor(0.0, device=device)
        
        # PDE损失
        if ch_logits.pde is not None:
            # 正确处理PDE logits的形状：[batch, bins, seq, seq] -> [batch, seq, seq, bins]
            pde_logits_reordered = ch_logits.pde.permute(0, 2, 3, 1)  # [batch, seq, seq, bins]
            pde_logits_flat = pde_logits_reordered.reshape(-1, pde_logits_reordered.shape[-1])  # [batch*seq*seq, bins]
            pde_labels_flat = pde_labels.reshape(-1)  # [batch*seq*seq]
            
            # 处理标签中的越界值
            num_pde_bins = pde_logits_flat.shape[-1]
            pde_labels_flat = torch.clamp(pde_labels_flat, min=0, max=num_pde_bins-1)
            
            pde_loss = F.cross_entropy(
                pde_logits_flat,
                pde_labels_flat,
                ignore_index=ignore_index,
                reduction='mean'
            )
        else:
            pde_loss = torch.tensor(0.0, device=device)
        
        # pLDDT损失
        if ch_logits.plddt is not None:
            # pLDDT logits的形状通常是[batch, seq, bins]，应该与[batch, seq]的标签匹配
            plddt_logits_flat = ch_logits.plddt.reshape(-1, ch_logits.plddt.shape[-1])  # [batch*seq, bins]
            plddt_labels_flat = plddt_labels.reshape(-1)  # [batch*seq]
            
            # 处理标签中的越界值
            num_plddt_bins = ch_logits.plddt.shape[-1]
            plddt_labels_flat = torch.clamp(plddt_labels_flat, min=0, max=num_plddt_bins-1)
            
            # 如果形状仍然不匹配，使用最小尺寸
            if plddt_logits_flat.shape[0] != plddt_labels_flat.shape[0]:
                min_size = min(plddt_logits_flat.shape[0], plddt_labels_flat.shape[0])
                plddt_logits_flat = plddt_logits_flat[:min_size]
                plddt_labels_flat = plddt_labels_flat[:min_size]
            
            plddt_loss = F.cross_entropy(
                plddt_logits_flat,
                plddt_labels_flat,
                ignore_index=ignore_index,
                reduction='mean'
            )
        else:
            plddt_loss = torch.tensor(0.0, device=device)
        
        # Distogram损失
        if distogram_logits is not None:
            # distogram logits的形状通常是[batch, bins, seq, seq]，需要重新排列
            distogram_logits_reordered = distogram_logits.permute(0, 2, 3, 1)  # [batch, seq, seq, bins]
            distogram_logits_flat = distogram_logits_reordered.reshape(-1, distogram_logits_reordered.shape[-1])
            distogram_labels_flat = distogram_labels.reshape(-1)
            
            # 处理标签中的越界值
            num_distogram_bins = distogram_logits_flat.shape[-1]
            distogram_labels_flat = torch.clamp(distogram_labels_flat, min=0, max=num_distogram_bins-1)
            
            distogram_loss = F.cross_entropy(
                distogram_logits_flat,
                distogram_labels_flat,
                ignore_index=ignore_index,
                reduction='mean'
            )
        else:
            distogram_loss = torch.tensor(0.0, device=device)
        
        # Resolved损失（这里简化处理）
        resolved_loss = torch.mean(ch_logits.resolved * 0.0)
        
        # 总置信度损失
        total_confidence_loss = self.loss_pae_weight * pae_loss + pde_loss + plddt_loss
        
        confidence_results = {
            'pae_loss': pae_loss,
            'pde_loss': pde_loss,
            'plddt_loss': plddt_loss,
            'resolved_loss': resolved_loss,
            'distogram_loss': distogram_loss,
            'logits': ch_logits,
            'distogram_logits': distogram_logits
        }
        
        return total_confidence_loss, confidence_results
    
    def _compute_pae_labels(self, pred_coords, true_coords, molecule_atom_indices, mask):
        """计算PAE标签"""
        device = pred_coords.device
        batch_size = pred_coords.shape[0]
        

        # 处理molecule_atom_indices中的-1值
        # 将-1替换为0，并创建有效索引的mask
        valid_indices_mask = molecule_atom_indices >= 0
        safe_indices = torch.where(valid_indices_mask, molecule_atom_indices, 0)
        
        # 提取分子位置
        molecule_pos = pred_coords.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        true_molecule_pos = true_coords.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # 对无效位置进行mask处理
        valid_mask = valid_indices_mask.unsqueeze(-1).expand(-1, -1, 3)
        molecule_pos = molecule_pos * valid_mask.float()
        true_molecule_pos = true_molecule_pos * valid_mask.float()
        
        # 计算对齐误差
        align_error = torch.cdist(molecule_pos, true_molecule_pos, p=2)
        
        # 转换为bins
        pae_labels = distance_to_dgram(align_error, self.pae_bins, return_labels=True)
        
        # 应用mask
        pair_mask = to_pairwise_mask(mask)
        # 同时考虑有效索引mask
        valid_pair_mask = valid_indices_mask.unsqueeze(-1) & valid_indices_mask.unsqueeze(-2)
        final_mask = pair_mask & valid_pair_mask
        pae_labels = torch.where(final_mask, pae_labels, self.ignore_index)
        
        return pae_labels
    
    def _compute_pde_labels(self, pred_coords, true_coords, molecule_atom_indices, mask):
        """计算PDE标签"""
        device = pred_coords.device
        
        # 处理molecule_atom_indices中的-1值
        # 将-1替换为0，并创建有效索引的mask
        valid_indices_mask = molecule_atom_indices >= 0
        safe_indices = torch.where(valid_indices_mask, molecule_atom_indices, 0)
        
        # 提取分子位置
        molecule_pos = pred_coords.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        true_molecule_pos = true_coords.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # 对无效位置进行mask处理
        valid_mask = valid_indices_mask.unsqueeze(-1).expand(-1, -1, 3)
        molecule_pos = molecule_pos * valid_mask.float()
        true_molecule_pos = true_molecule_pos * valid_mask.float()
        
        # 计算距离
        pred_dist = torch.cdist(molecule_pos, molecule_pos, p=2)
        true_dist = torch.cdist(true_molecule_pos, true_molecule_pos, p=2)
        
        # 计算距离误差
        dist_error = torch.abs(pred_dist - true_dist)
        
        # 转换为bins
        pde_labels = distance_to_dgram(dist_error, self.pde_bins, return_labels=True)
        
        # 应用mask
        pair_mask = to_pairwise_mask(mask)
        # 同时考虑有效索引mask
        valid_pair_mask = valid_indices_mask.unsqueeze(-1) & valid_indices_mask.unsqueeze(-2)
        final_mask = pair_mask & valid_pair_mask
        pde_labels = torch.where(final_mask, pde_labels, self.ignore_index)
        
        return pde_labels
    
    def _compute_plddt_labels(self, pred_coords, true_coords, is_molecule_types, molecule_atom_lens, atom_mask):
        """计算pLDDT标签"""
        device = pred_coords.device
        
        # 计算距离
        pred_dists = torch.cdist(pred_coords, pred_coords, p=2)
        true_dists = torch.cdist(true_coords, true_coords, p=2)
        
        # 创建inclusion mask
        is_rna = batch_repeat_interleave(is_molecule_types[..., IS_RNA_INDEX], molecule_atom_lens)
        is_dna = batch_repeat_interleave(is_molecule_types[..., IS_DNA_INDEX], molecule_atom_lens)
        is_nucleotide = is_rna | is_dna
        
        is_any_nucleotide_pair = is_nucleotide.unsqueeze(-1) | is_nucleotide.unsqueeze(-2)
        
        inclusion_radius = torch.where(
            is_any_nucleotide_pair,
            true_dists < self.lddt_mask_nucleic_acid_cutoff,
            true_dists < self.lddt_mask_other_cutoff,
        )
        
        # 计算LDDT
        dist_diff = torch.abs(true_dists - pred_dists)
        # 使用预定义的LDDT阈值
        lddt_thresholds = torch.tensor(self.lddt_threshold_values, device=device)
        lddt = (dist_diff.unsqueeze(-1) < lddt_thresholds).float().mean(dim=-1)
        
        # 应用mask
        plddt_mask = inclusion_radius & to_pairwise_mask(atom_mask)
        plddt_mask = plddt_mask & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=device)
        
        # 计算平均LDDT
        lddt_mean = masked_average(lddt, plddt_mask, dim=-1)
        
        # 转换为bins
        plddt_labels = torch.clamp(
            torch.floor(lddt_mean * self.num_plddt_bins).long(),
            max=self.num_plddt_bins - 1
        )
        
        return plddt_labels
    
    def _compute_distogram_labels(self, pred_coords, true_coords, molecule_atom_indices, mask):
        """计算distogram标签"""
        device = pred_coords.device
        
        # 处理molecule_atom_indices中的-1值
        valid_indices_mask = molecule_atom_indices >= 0
        safe_indices = torch.where(valid_indices_mask, molecule_atom_indices, 0)
        
        # 提取分子位置
        molecule_pos = pred_coords.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        true_molecule_pos = true_coords.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # 对无效位置进行mask处理
        valid_mask = valid_indices_mask.unsqueeze(-1).expand(-1, -1, 3)
        true_molecule_pos = true_molecule_pos * valid_mask.float()
        
        # 计算真实距离
        true_distances = torch.cdist(true_molecule_pos, true_molecule_pos, p=2)
        
        # 转换为bins
        distogram_labels = distance_to_dgram(true_distances, self.distance_bins, return_labels=True)
        
        # 应用mask
        pair_mask = to_pairwise_mask(mask)
        valid_pair_mask = valid_indices_mask.unsqueeze(-1) & valid_indices_mask.unsqueeze(-2)
        final_mask = pair_mask & valid_pair_mask
        distogram_labels = torch.where(final_mask, distogram_labels, self.ignore_index)
        
        return distogram_labels
    


    def set_train_mode(self):
        """设置为训练模式"""
        self.train()
        # 确保RhoFold始终保持eval模式
        self.rhofold.eval()
    
    def set_eval_mode(self):
        """设置为推理模式"""
        self.eval()
    
    def forward(self, tokens, rna_fm_tokens, seq, target_coords=None, missing_atom_mask=None, seq_lengths=None, **kwargs):
        # 确保RhoFold前向传播时不计算梯度
        batch_num = tokens.shape[0]
        is_batched = batch_num > 1
        try:
            with torch.no_grad():
                logger.debug(f"RhoFold输入: {tokens.shape} {rna_fm_tokens.shape if rna_fm_tokens is not None else 'None'}")
                outputs, single_fea, pair_fea = self.rhofold(tokens, rna_fm_tokens, seq[0], **kwargs)
        except Exception as e:
            logger.error(f"⚠️ RhoFold前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            exit()
                
        # 如果需要梯度用于后续计算，可以detach并重新设置requires_grad
        if single_fea is not None:
            single_fea = single_fea.detach()
            # 使用维度适配器调整single_fea从256维到384维
            single_fea = self.single_dim_adapter(single_fea)
        if pair_fea is not None:
            pair_fea = pair_fea.detach()
            # 为pair_fea添加Layer Normalization以提高稳定性
            # pair_fea: [bs, seq_len, seq_len, 128]
            pair_fea = F.layer_norm(pair_fea, pair_fea.shape[-1:])
            
        # single_fea: [bs, seq_len, dim_single(384)] 
        # pair_fea: [bs, seq_len, seq_len, dim_pairwise(128)]

        # 如果没有目标坐标且在训练模式，返回RhoFold的输出用于测试
        if target_coords is None:
            if self.training:
                logger.warning("⚠️ 训练模式但没有目标坐标")
                return None, None, None
        
        # 处理target_coords，将batch张量转换为列表格式
        atom_pos_list = None
        af_in = None
        atom_mask = None
        if target_coords is not None:
            atom_pos_list = [target_coords[0]]
            af_in, atom_mask = process_alphafold3_input(
                ss_rna=[seq[0]],
                atom_pos=atom_pos_list,
            )
            logger.debug(f"AlphaFold3输入（对齐前）: "
                        f"atom_inputs {af_in.atom_inputs.shape if af_in.atom_inputs is not None else 'None'}, "
                        f"atompair_inputs {af_in.atompair_inputs.shape if af_in.atompair_inputs is not None else 'None'}, "
                        f"additional_token_feats {af_in.additional_token_feats.shape if af_in.additional_token_feats is not None else 'None'}, "
                        f"molecule_atom_lens {af_in.molecule_atom_lens.shape if af_in.molecule_atom_lens is not None else 'None'}, "
                        f"molecule_ids {af_in.molecule_ids.shape if af_in.molecule_ids is not None else 'None'}, "
                        f"atom_mask {atom_mask.shape if atom_mask is not None else 'None'}")
        else:
            logger.debug("推理模式：无目标坐标，跳过AlphaFold3输入处理")
            af_in, atom_mask = process_alphafold3_input(
                ss_rna=[seq[0]],
                atom_pos=None,
            )

        # 🔧 使用align_input函数将AlphaFold3格式转换为RhoFold+格式
        logger.debug("开始格式对齐：AlphaFold3 -> RhoFold+")
        af_in_aligned, aligned_atom_mask = align_input(af_in, seq)
        
        # 更新atom_mask为对齐后的版本
        atom_mask = aligned_atom_mask
        
        logger.debug(f"AlphaFold3输入（对齐后）: "
                    f"atom_inputs {af_in_aligned.atom_inputs.shape if af_in_aligned.atom_inputs is not None else 'None'}, "
                    f"atompair_inputs {af_in_aligned.atompair_inputs.shape if af_in_aligned.atompair_inputs is not None else 'None'}, "
                    f"additional_token_feats {af_in_aligned.additional_token_feats.shape if af_in_aligned.additional_token_feats is not None else 'None'}, "
                    f"molecule_atom_lens {af_in_aligned.molecule_atom_lens.shape if af_in_aligned.molecule_atom_lens is not None else 'None'}, "
                    f"molecule_ids {af_in_aligned.molecule_ids.shape if af_in_aligned.molecule_ids is not None else 'None'}, "
                    f"atom_mask {atom_mask.shape if atom_mask is not None else 'None'}")
        
        # 使用对齐后的af_in
        af_in = af_in_aligned

        # 基于缺失原子掩码/有效原子掩码进行按 batch 的质心归零
        if af_in.atom_pos is not None:
            try:
                pos = af_in.atom_pos  # [B, N, 3]
                # 优先使用用户提供的 missing_atom_mask（True=缺失，取反为有效）
                mask_src = None
                mask_src = (~missing_atom_mask).to(pos.dtype)  # [B, N]

                mask_b = mask_src.unsqueeze(-1)  # [B, N, 1]
                valid_counts = mask_b.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
                centroid = (pos * mask_b).sum(dim=1, keepdim=True) / valid_counts  # [B, 1, 3]
                af_in.atom_pos = pos - centroid
            except Exception as e:
                logger.warning(f"质心归零时出现异常，跳过归零：{e}")

        # 获取设备信息并将所有输入数据移动到正确的设备
        device = tokens.device
        
        # 将af_in的所有属性移动到正确的设备
        if af_in.atom_inputs is not None:
            af_in.atom_inputs = af_in.atom_inputs.to(device)
        if af_in.atompair_inputs is not None:
            af_in.atompair_inputs = af_in.atompair_inputs.to(device)
        if atom_mask is not None:
            atom_mask = atom_mask.to(device)
        if af_in.additional_token_feats is not None:
            af_in.additional_token_feats = af_in.additional_token_feats.to(device)
        if af_in.molecule_atom_lens is not None:
            af_in.molecule_atom_lens = af_in.molecule_atom_lens.to(device)
        if af_in.molecule_ids is not None:
            af_in.molecule_ids = af_in.molecule_ids.to(device)
        if af_in.atom_parent_ids is not None:
            af_in.atom_parent_ids = af_in.atom_parent_ids.to(device)
        if missing_atom_mask is not None:
            missing_atom_mask = missing_atom_mask.to(device)
        if af_in.additional_molecule_feats is not None:
            af_in.additional_molecule_feats = af_in.additional_molecule_feats.to(device)
        if af_in.is_molecule_types is not None:
            af_in.is_molecule_types = af_in.is_molecule_types.to(device)
        if af_in.molecule_atom_indices is not None:
            af_in.molecule_atom_indices = af_in.molecule_atom_indices.to(device)
        if hasattr(af_in, 'token_bonds') and af_in.token_bonds is not None:
            af_in.token_bonds = af_in.token_bonds.to(device)

        (
            single_inputs,
            _,
            _,
            atom_feats,
            atompair_feats

        ) = self.input_embedder(
            atom_inputs = af_in.atom_inputs,
            atompair_inputs = af_in.atompair_inputs,
            atom_mask = atom_mask,
            additional_token_feats = af_in.additional_token_feats,
            molecule_atom_lens = af_in.molecule_atom_lens,
            molecule_ids = af_in.molecule_ids
        )

        # 生成更精确的mask，考虑序列长度的差异
        if af_in.molecule_atom_lens is not None:
            mask = af_in.molecule_atom_lens > 0
            
            # 如果提供了seq_lengths，进一步细化mask以处理序列长度差异
            if seq_lengths is not None and is_batched:
                # 基于真实序列长度调整单个表示的mask
                batch_size, max_seq_len = single_fea.shape[0], single_fea.shape[1]
                seq_mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.bool)
                
                for i in range(batch_size):
                    actual_len = seq_lengths[i].item()
                    seq_mask[i, :actual_len] = True
                
                logger.debug(f"生成序列mask: {seq_mask.shape}, 真实长度: {seq_lengths.tolist() if seq_lengths is not None else 'None'}")
                
                # 将seq_mask应用到single和pair特征上
                single_fea = single_fea * seq_mask.unsqueeze(-1)  # [bs, seq_len, dim] * [bs, seq_len, 1]
                pair_fea = pair_fea * seq_mask.unsqueeze(-1).unsqueeze(-2)  # 应用到行
                pair_fea = pair_fea * seq_mask.unsqueeze(-2).unsqueeze(-1)  # 应用到列
        else:
            logger.error("没有提供molecule_atom_lens")
            exit()

        relative_position_encoding = self.relative_position_encoding(
            additional_molecule_feats = af_in.additional_molecule_feats,
        )
        
        # 判断是否有ground truth数据
        has_ground_truth = target_coords is not None and af_in.atom_pos is not None
        
        if has_ground_truth:
            # 有ground truth：计算损失（无论training还是eval模式）
            logger.debug(f"计算损失 - 模式: {'训练' if self.training else '验证'}")
            
            # 🔍 验证batch数据一致性（在训练模式下）
            if self.mask_validator_on and self.training and hasattr(self, 'mask_validator') and missing_atom_mask is not None:
                try: 
                    # 只有在有必要数据时才进行验证
                    if target_coords is not None and seq is not None:
                        batch_validation = self.mask_validator.validate_batch_consistency(
                            tokens=tokens,
                            sequences=seq,
                            coordinates=target_coords,
                            missing_atom_mask=missing_atom_mask,
                            seq_lengths=seq_lengths
                        )
                        
                        # 验证EDM输入的mask一致性
                        edm_validation = self.mask_validator.validate_edm_inputs(
                            atom_mask=atom_mask,
                            missing_atom_mask=missing_atom_mask,
                            molecule_atom_lens=af_in.molecule_atom_lens,
                            mask=mask
                        )
                        
                        # 如果有严重问题，记录警告
                        if not batch_validation['is_valid'] or not edm_validation['is_valid']:
                            logger.warning(f"⚠️ Mask验证发现问题:")
                            for error in batch_validation.get('errors', []) + edm_validation.get('errors', []):
                                logger.warning(f"  - {error}")
                            
                            # 在严格模式下可能会抛出异常，这里只记录
                            if len(batch_validation.get('errors', [])) > 0:
                                logger.warning(f"  batch验证错误数: {len(batch_validation['errors'])}")
                            if len(edm_validation.get('errors', [])) > 0:
                                logger.warning(f"  EDM输入验证错误数: {len(edm_validation['errors'])}")
                        
                        # 记录统计信息（可选）
                        if batch_validation.get('statistics'):
                            stats = batch_validation['statistics']
                            if 'seq_length_variance' in stats and stats['seq_length_variance'] > 0:
                                logger.debug(f"  序列长度差异: {stats['seq_length_variance']}")
                            if 'avg_atom_coverage' in stats:
                                logger.debug(f"  平均原子覆盖率: {stats['avg_atom_coverage']:.2%}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Mask验证过程出错: {e}")
                    # 验证失败不应该影响训练，继续执行
            
            diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm(
                atom_pos_ground_truth = af_in.atom_pos,
                additional_molecule_feats = af_in.additional_molecule_feats,
                is_molecule_types = af_in.is_molecule_types,
                add_smooth_lddt_loss = False,
                add_bond_loss = False,
                atom_feats = atom_feats,
                atompair_feats = atompair_feats,
                atom_parent_ids = af_in.atom_parent_ids,
                missing_atom_mask = missing_atom_mask,
                atom_mask = atom_mask,
                mask = mask,
                single_trunk_repr = single_fea,
                single_inputs_repr = single_inputs,
                pairwise_trunk = pair_fea,
                pairwise_rel_pos_feats = relative_position_encoding,
                molecule_atom_lens = af_in.molecule_atom_lens,
                molecule_atom_indices = af_in.molecule_atom_indices,
                token_bonds = af_in.token_bonds,
                return_denoised_pos = True,
                nucleotide_loss_weight = self.nucleotide_loss_weight,
                ligand_loss_weight = self.ligand_loss_weight,
            )
            
            # 计算置信度损失（使用修复的molecule_atom_indices）
            try:
                # 预处理molecule_atom_indices，将-1替换为0以避免索引越界
                if af_in.molecule_atom_indices is not None:
                    safe_molecule_atom_indices = af_in.molecule_atom_indices.clone()
                    safe_molecule_atom_indices = torch.where(
                        safe_molecule_atom_indices >= 0,
                        safe_molecule_atom_indices,
                        0
                    )
                else:
                    safe_molecule_atom_indices = None
                
                # 调用完整的置信度损失计算，使用安全的索引
                confidence_loss, confidence_logits = self._compute_confidence_loss(
                    single_fea, single_inputs, pair_fea, denoised_atom_pos, atom_feats,
                    af_in.atom_pos, safe_molecule_atom_indices, af_in.molecule_atom_lens,
                    af_in.is_molecule_types, af_in.additional_molecule_feats, mask, atom_mask
                )
                
            except Exception as e:
                logger.warning(f"WARNING: 置信度损失计算失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 回退到简单损失
                confidence_loss = torch.tensor(0.0, device=diffusion_loss.device)
                confidence_logits = {
                    'pae_loss': torch.tensor(0.0, device=diffusion_loss.device),
                    'pde_loss': torch.tensor(0.0, device=diffusion_loss.device),
                    'plddt_loss': torch.tensor(0.0, device=diffusion_loss.device),
                    'resolved_loss': torch.tensor(0.0, device=diffusion_loss.device),
                    'logits': None
                }
            
            # 计算总损失
            distogram_loss = confidence_logits.get('distogram_loss', torch.tensor(0.0, device=diffusion_loss.device))
            total_loss = (diffusion_loss * self.loss_diffusion_weight + 
                         confidence_loss * self.loss_confidence_weight + 
                         distogram_loss * self.loss_distogram_weight)
            
            # 创建详细的损失分解
            zero_tensor = torch.tensor(0.0, device=diffusion_loss.device)
            loss_breakdown = DiffoldLossBreakdown(
                total_loss=total_loss,
                total_diffusion=diffusion_loss,
                distogram=distogram_loss,
                pae=confidence_logits.get('pae_loss', zero_tensor),
                pde=confidence_logits.get('pde_loss', zero_tensor),
                plddt=confidence_logits.get('plddt_loss', zero_tensor),
                resolved=confidence_logits.get('resolved_loss', zero_tensor),
                confidence=confidence_loss,
                diffusion_mse=diffusion_loss_breakdown.diffusion_mse,
                diffusion_bond=diffusion_loss_breakdown.diffusion_bond,
                diffusion_smooth_lddt=diffusion_loss_breakdown.diffusion_smooth_lddt,
            )
            
            return {
                'loss': total_loss,
                'predicted_coords': denoised_atom_pos,
                'loss_breakdown': loss_breakdown,
                'confidence_logits': confidence_logits.get('logits', None),
                'distogram_logits': confidence_logits.get('distogram_logits', None),
                'atom_mask': atom_mask,
                'mode': 'supervised'
            }
        else:
            # 没有ground truth：进行采样
            logger.info("进行采样生成")
            
            sampled_coords = self.edm.sample(
                atom_feats = atom_feats,
                atompair_feats = atompair_feats,
                atom_parent_ids = af_in.atom_parent_ids,
                atom_mask = atom_mask,
                mask = mask,
                single_trunk_repr = single_fea,
                single_inputs_repr = single_inputs,
                pairwise_trunk = pair_fea,
                pairwise_rel_pos_feats = relative_position_encoding,
                molecule_atom_lens = af_in.molecule_atom_lens,
                num_sample_steps = 32,  # 采样步数
                use_tqdm_pbar = False,
            )
            
            return {
                'predicted_coords': sampled_coords,
                'atom_mask': atom_mask,
                'mode': 'sampling'
            }

