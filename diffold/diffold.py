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
from diffold.input_processor import process_alphafold3_input

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

    def forward(self, tokens, rna_fm_tokens, seq, target_coords=None, missing_atom_mask=None, **kwargs):
        # 确保RhoFold前向传播时不计算梯度
        try:
            with torch.no_grad():
                outputs, single_fea, pair_fea = self.rhofold(tokens, rna_fm_tokens, seq, **kwargs)
        except Exception as e:
            print(f"⚠️ RhoFold前向传播失败: {e}")
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
                print("⚠️ 训练模式但没有目标坐标，返回模拟损失")
                # 返回一个模拟的损失用于测试
                dummy_loss = torch.tensor(1.0, device=tokens.device, requires_grad=True)
                return dummy_loss, None, None

        # 如果有目标坐标，继续diffusion训练
        af_in, atom_mask = process_alphafold3_input(
            ss_rna=[seq],
            atom_pos=target_coords,
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
            af_in.missing_atom_mask = missing_atom_mask.to(device)
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

        if self.training:
            mask = af_in.molecule_atom_lens > 0
            relative_position_encoding = self.relative_position_encoding(
                additional_molecule_feats = af_in.additional_molecule_feats,
            )
            
            diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm(
                atom_pos_ground_truth = target_coords,
                additional_molecule_feats = af_in.additional_molecule_feats,
                is_molecule_types = af_in.is_molecule_types,
                add_smooth_lddt_loss = False,
                add_bond_loss = False,
                atom_feats = atom_feats,
                atompair_feats = atompair_feats,
                atom_parent_ids = af_in.atom_parent_ids,
                missing_atom_mask = af_in.missing_atom_mask,
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
                verbose = True,
            )
            return diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown
        else:
            pass

