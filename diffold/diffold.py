from torch import nn
import torch
import os

from alphafold3_pytorch import(
    InputFeatureEmbedder,
    ElucidatedAtomDiffusion,
    DiffusionModule,
    RelativePositionEncoding,
)
from alphafold3_pytorch.inputs import NUM_MOLECULE_IDS
from diffold.input_processor import process_alphafold3_input, process_multiple_alphafold3_inputs

from rhofold.rhofold import RhoFold

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

        self.rhofold = RhoFold(config)
        
        # 加载RhoFold预训练权重
        if rhofold_checkpoint_path is None:
            rhofold_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained', 'model_20221010_params.pt')
        
        if os.path.exists(rhofold_checkpoint_path):
            print(f"正在加载RhoFold预训练权重: {rhofold_checkpoint_path}")
            checkpoint = torch.load(rhofold_checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                self.rhofold.load_state_dict(checkpoint['model'])
            else:
                self.rhofold.load_state_dict(checkpoint)
            print("✓ RhoFold预训练权重加载成功")
        else:
            print(f"⚠ 警告: 未找到预训练权重文件: {rhofold_checkpoint_path}")
        
        # 固定RhoFold模型参数，不参与训练
        for param in self.rhofold.parameters():
            param.requires_grad = False
        
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
            token_transformer_depth=24,
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
        # RhoFold输出: single_fea [bs, seq_len, 256], pair_fea [bs, seq_len, seq_len, 128]
        # diffusion期望: single_fea [bs, seq_len, 384], pair_fea [bs, seq_len, seq_len, 128]
        self.single_dim_adapter = nn.Linear(256, 384)  # 将256维调整到384维

    def freeze_rhofold(self):
        """冻结RhoFold模型参数"""
        for param in self.rhofold.parameters():
            param.requires_grad = False
        self.rhofold.eval()
    
    def unfreeze_rhofold(self):
        """解冻RhoFold模型参数"""
        for param in self.rhofold.parameters():
            param.requires_grad = True
        self.rhofold.train()
    
    def get_trainable_parameters(self):
        """获取可训练的参数（排除RhoFold）"""
        trainable_params = []
        for name, param in self.named_parameters():
            if not name.startswith('rhofold.') and param.requires_grad:
                trainable_params.append(param)
        return trainable_params

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
            if is_batched:
                # 获取序列长度信息
                if seq_lengths is None:
                    # 如果没有提供seq_lengths，尝试从seq中推断
                    if isinstance(seq, list):
                        seq_lengths = torch.tensor([len(s) for s in seq], device=tokens.device)
                    else:
                        # 假设所有序列都是最大长度（保持原有行为）
                        seq_lengths = torch.full((batch_num,), tokens.shape[1], device=tokens.device)
                
                # 准备存储结果的列表
                single_features = []
                pair_features = []
                max_seq_len = tokens.shape[2]  # padding后的最大长度
                
                print(f"处理batch: {batch_num}个序列, 最大长度: {max_seq_len}")
                
                for i in range(batch_num):
                    if seq_lengths is not None:
                        current_seq_len = seq_lengths[i].item()
                    else:
                        current_seq_len = tokens.shape[2]  # 使用padding后的最大长度
                    print(f"处理序列 {i+1}/{batch_num}, 真实长度: {current_seq_len}")
                    
                    # 去掉padding，只保留真实序列部分
                    current_tokens = tokens[i:i+1, :, :current_seq_len]  # [1, MSA_depth, real_len]
                    current_rna_fm_tokens = rna_fm_tokens[i:i+1, :current_seq_len] if rna_fm_tokens is not None else None
                    
                    current_seq = seq[i]
                    
                    # 处理当前序列
                    with torch.no_grad():
                        print(f"RhoFold输入[{i}]: tokens {current_tokens.shape}, rna_fm_tokens {current_rna_fm_tokens.shape if current_rna_fm_tokens is not None else 'None'}")
                        outputs, single, pair = self.rhofold(current_tokens, current_rna_fm_tokens, current_seq, **kwargs)
                        
                        # single: [1, real_len, 256], pair: [1, real_len, real_len, 128]
                        print(f"RhoFold输出[{i}]: single {single.shape}, pair {pair.shape}")
                        
                        # 将结果padding回原始大小以便后续batch处理
                        padded_single = torch.zeros(1, max_seq_len, single.shape[-1], device=tokens.device)
                        padded_pair = torch.zeros(1, max_seq_len, max_seq_len, pair.shape[-1], device=tokens.device)
                        
                        padded_single[:, :current_seq_len] = single
                        padded_pair[:, :current_seq_len, :current_seq_len] = pair
                        
                        single_features.append(padded_single)
                        pair_features.append(padded_pair)
                
                # 将所有序列的结果重新组合成batch
                single_fea = torch.cat(single_features, dim=0)  # [batch_num, max_seq_len, 256]
                pair_fea = torch.cat(pair_features, dim=0)      # [batch_num, max_seq_len, max_seq_len, 128]
                
                print(f"合并后的特征: single {single_fea.shape}, pair {pair_fea.shape}")
                
            else:
                with torch.no_grad():
                    print("RhoFold输入:", tokens.shape, rna_fm_tokens.shape if rna_fm_tokens is not None else 'None')
                    outputs, single_fea, pair_fea = self.rhofold(tokens, rna_fm_tokens, seq, **kwargs)
        except Exception as e:
            print(f"⚠️ RhoFold前向传播失败: {e}")
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
            
        # single_fea: [bs, seq_len, dim_single(384)] 
        # pair_fea: [bs, seq_len, seq_len, dim_pairwise(128)]

        # 如果没有目标坐标且在训练模式，返回RhoFold的输出用于测试
        if target_coords is None:
            if self.training:
                print("⚠️ 训练模式但没有目标坐标")
                return None, None, None

        print("RhoFold输出:", single_fea.shape, pair_fea.shape)

        
        # 处理target_coords，将batch张量转换为列表格式
        atom_pos_list = None
        if target_coords is not None:
            if is_batched:
                # 为batch处理准备输入列表
                input_list = []
                for i in range(batch_num):
                    if seq_lengths is not None:
                        current_seq_len = seq_lengths[i].item()
                    else:
                        current_seq_len = tokens.shape[2]  # 使用padding后的最大长度
                    
                    # 获取当前序列的坐标
                    if missing_atom_mask is not None:
                        # 使用missing_atom_mask确定真实原子数量
                        current_missing_mask = missing_atom_mask[i]  # type: ignore
                        # 找到有效原子的数量
                        valid_atoms = (~current_missing_mask).sum().item()
                        if valid_atoms > 0:
                            current_coords = target_coords[i, :valid_atoms]  # [valid_atoms, 3]
                        else:
                            # 如果没有有效原子，创建空张量
                            current_coords = torch.empty(0, 3, device=target_coords.device)
                    else:
                        # 如果没有missing_atom_mask，使用所有坐标
                        current_coords = target_coords[i]  # [max_atoms, 3]
                    
                    # 为每个序列创建单独的输入字典
                    input_dict = {
                        'ss_rna': [seq[i]],  # type: ignore # 单个序列作为列表
                        'atom_pos': [current_coords]  # 单个坐标张量作为列表
                    }
                    input_list.append(input_dict)
                    
                    print(f"序列[{i}]: {seq[i][:10] if seq is not None else 'None'}..., 序列长度: {current_seq_len}, 原子数: {current_coords.shape[0]}")
                
                # 使用process_multiple_alphafold3_inputs处理batch
                result = process_multiple_alphafold3_inputs(
                    input_list, 
                )
                af_in, atom_mask = result  # type: ignore
            else:
                # 单个序列的情况，使用原来的方式
                if missing_atom_mask is not None:
                    valid_atoms = (~missing_atom_mask[0]).sum().item()
                    if valid_atoms > 0:
                        atom_pos_list = [target_coords[0, :valid_atoms]]
                    else:
                        atom_pos_list = [torch.empty(0, 3, device=target_coords.device)]
                else:
                    atom_pos_list = [target_coords[0]]
                
                af_in, atom_mask = process_alphafold3_input(
                    ss_rna=[seq[0]] if seq is not None else [],
                    atom_pos=atom_pos_list,
                )
        else:
            print("没有目标坐标")
            exit()

        print("AlphaFold3输入:",
            "atom_inputs", af_in.atom_inputs.shape if af_in.atom_inputs is not None else 'None',
            "atompair_inputs", af_in.atompair_inputs.shape if af_in.atompair_inputs is not None else 'None',
            "additional_token_feats", af_in.additional_token_feats.shape if af_in.additional_token_feats is not None else 'None',
            "molecule_atom_lens", af_in.molecule_atom_lens.shape if af_in.molecule_atom_lens is not None else 'None',
            "molecule_ids", af_in.molecule_ids.shape if af_in.molecule_ids is not None else 'None',
            "atom_mask", atom_mask.shape if atom_mask is not None else 'None'
        )

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
                
                print(f"生成序列mask: {seq_mask.shape}, 真实长度: {seq_lengths.tolist() if seq_lengths is not None else 'None'}")
                
                # 将seq_mask应用到single和pair特征上
                single_fea = single_fea * seq_mask.unsqueeze(-1)  # [bs, seq_len, dim] * [bs, seq_len, 1]
                pair_fea = pair_fea * seq_mask.unsqueeze(-1).unsqueeze(-2)  # 应用到行
                pair_fea = pair_fea * seq_mask.unsqueeze(-2).unsqueeze(-1)  # 应用到列
        else:
            print("没有提供molecule_atom_lens")
            exit()

        # 处理missing_atom_mask的维度对齐
        if af_in.atom_pos is not None and missing_atom_mask is not None:
            batch_num = af_in.atom_pos.shape[0]
            padded_atom_num = af_in.atom_pos.shape[1]
            real_atom_num = missing_atom_mask.shape[1]
            if padded_atom_num > real_atom_num:    
                missing_atom_mask = torch.cat([
                    missing_atom_mask, 
                    torch.ones(batch_num, padded_atom_num - real_atom_num, device=device, dtype=missing_atom_mask.dtype)
                ], dim=1)
                print(f"扩展missing_atom_mask: {real_atom_num} -> {padded_atom_num}")

        relative_position_encoding = self.relative_position_encoding(
            additional_molecule_feats = af_in.additional_molecule_feats,
        )
        
        # 判断是否有ground truth数据
        has_ground_truth = target_coords is not None and af_in.atom_pos is not None
        
        if has_ground_truth:
            # 有ground truth：计算损失（无论training还是eval模式）
            print(f"计算损失 - 模式: {'训练' if self.training else '验证'}")
            
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
            
            return {
                'loss': diffusion_loss,
                'predicted_coords': denoised_atom_pos,
                'loss_breakdown': diffusion_loss_breakdown,
                'atom_mask': atom_mask,
                'mode': 'supervised'
            }
        else:
            # 没有ground truth：进行采样
            print("进行采样生成")
            
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

