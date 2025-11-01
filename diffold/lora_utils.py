"""
LoRA (Low-Rank Adaptation) 工具模块
用于高效微调Diffold模型
"""

import logging
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        TaskType
    )
    PEFT_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ PEFT库未安装，LoRA功能将不可用。请运行: uv pip install peft")
    PEFT_AVAILABLE = False


class LoRAManager:
    """LoRA管理器，负责配置和应用LoRA到Diffold模型"""
    
    def __init__(self, config: Dict):
        """
        初始化LoRA管理器
        
        Args:
            config: LoRA配置字典，包含:
                - enable: 是否启用LoRA
                - r: LoRA秩
                - alpha: LoRA缩放因子
                - dropout: LoRA dropout率
                - target_modules: 目标模块列表
                - strategy: LoRA应用策略
        """
        self.config = config
        self.enabled = config.get('enable', False)
        
        if self.enabled and not PEFT_AVAILABLE:
            raise ImportError(
                "启用了LoRA但PEFT库未安装。"
                "请运行: uv pip install peft>=0.7.0"
            )
    
    def get_target_modules(self, strategy: str = 'diffusion_only') -> List[str]:
        """
        根据策略获取LoRA目标模块
        
        Args:
            strategy: LoRA策略
                - 'diffusion_only': 仅扩散模块（推荐<100样本）
                - 'diffusion_confidence': 扩散+置信度头（推荐100-500样本）
                - 'diffusion_all_heads': 扩散+所有头部（推荐500-1000样本）
                - 'full_model': 全模型LoRA（推荐>1000样本）
                - 'custom': 自定义模块列表
        
        Returns:
            目标模块名称列表（匹配Linear层的名称模式）
        """
        # 🔥 PEFT只支持直接匹配Linear层名称，不能匹配父模块
        # 我们需要匹配alphafold3_pytorch中常见的Linear层命名模式
        
        # 基础扩散模块目标（优先级最高）
        # 匹配DiffusionTransformer, AtomEncoder, AtomDecoder中的注意力层
        diffusion_targets = [
            # 注意力层的Q/K/V投影
            "to_q",
            "to_k", 
            "to_v",
            "to_kv",  # 有些实现将k和v合并
            "to_out",  # 输出投影
            # 注意: 不包含to_gates，因为它通常是Sequential(Linear+Sigmoid)，PEFT不支持
        ]
        
        # 置信度头目标
        confidence_targets = [
            # Pairformer中的注意力层也使用相同的命名
            # 已经被上面的to_q, to_k等覆盖了
        ]
        
        # Distogram头目标  
        distogram_targets = [
            # Distogram头中的Linear层
        ]
        
        # RhoFold目标（仅在full_model时使用）
        rhofold_targets = [
            # RhoFold中的注意力层命名
            "linear_q",
            "linear_k",
            "linear_v", 
            "linear_o",
            "linear_g",
            "linear_kv",
        ]
        
        # 适配器目标
        adapter_targets = [
            "single_dim_adapter.0",  # 适配器中的第一个Linear层
        ]
        
        if strategy == 'diffusion_only':
            # 仅扩散模块的注意力层
            targets = diffusion_targets
        elif strategy == 'diffusion_confidence':
            # 扩散+置信度（置信度也用相同的to_q/k/v命名）
            targets = diffusion_targets
        elif strategy == 'diffusion_all_heads':
            # 扩散+所有头部
            targets = diffusion_targets
        elif strategy == 'full_model':
            # 全模型：包括RhoFold
            targets = list(set(diffusion_targets + rhofold_targets + adapter_targets))
        elif strategy == 'custom':
            # 使用配置中指定的自定义模块
            targets = self.config.get('custom_target_modules', diffusion_targets)
        else:
            logger.warning(f"未知的LoRA策略: {strategy}, 使用默认策略 'diffusion_only'")
            targets = diffusion_targets
        
        logger.info(f"🎯 LoRA策略 '{strategy}' 目标Linear层名称: {targets}")
        logger.info(f"   注意: PEFT将匹配所有名称中包含这些字符串的Linear层")
        return targets
    
    def _get_safe_linear_modules(self, model: nn.Module, strategy: str) -> List[str]:
        """
        获取安全的Linear模块列表（排除被Sequential包装的）
        
        PEFT无法处理被Sequential包装的Linear层，所以我们需要手动找出真正的nn.Linear模块
        """
        safe_modules = []
        target_patterns = self.get_target_modules(strategy)
        
        for name, module in model.named_modules():
            # 只选择纯nn.Linear，不是Sequential或其他容器
            if isinstance(module, nn.Linear):
                # 检查父模块是否是Sequential
                # 如果模块名包含目标模式，则添加
                if any(pattern in name for pattern in target_patterns):
                    # 额外检查：确保它不是Sequential的子模块
                    # 通过检查路径中是否有数字（Sequential的子模块通常是数字索引）
                    parent_name = '.'.join(name.split('.')[:-1])
                    try:
                        parent = model
                        for part in parent_name.split('.'):
                            if part:
                                parent = getattr(parent, part)
                        # 如果父模块不是Sequential，则这是安全的
                        if not isinstance(parent, nn.Sequential):
                            safe_modules.append(name)
                    except:
                        # 如果无法获取父模块，保守起见跳过
                        pass
        
        logger.info(f"🔍 找到 {len(safe_modules)} 个安全的Linear层用于LoRA")
        if safe_modules:
            logger.info(f"   示例: {safe_modules[:5]}")
        
        return safe_modules
    
    def create_lora_config(self, model: nn.Module = None) -> 'LoraConfig':
        """创建LoRA配置对象"""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT库未安装")
        
        # 获取目标模块
        strategy = self.config.get('strategy', 'diffusion_only')
        
        # 如果提供了模型，使用安全的模块列表
        if model is not None:
            target_modules = self._get_safe_linear_modules(model, strategy)
            if not target_modules:
                logger.warning("⚠️ 未找到安全的Linear层，回退到模式匹配")
                target_modules = self.get_target_modules(strategy)
        else:
            target_modules = self.get_target_modules(strategy)
        
        # LoRA超参数
        r = self.config.get('r', 8)
        lora_alpha = self.config.get('alpha', r)  # 默认等于r
        lora_dropout = self.config.get('dropout', 0.05)
        bias = self.config.get('bias', 'none')  # 'none', 'all', 'lora_only'
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.FEATURE_EXTRACTION,  # 结构预测任务
            inference_mode=False,
        )
        
        logger.info(f"📋 LoRA配置:")
        logger.info(f"  Rank (r): {r}")
        logger.info(f"  Alpha: {lora_alpha}")
        logger.info(f"  Dropout: {lora_dropout}")
        logger.info(f"  Bias: {bias}")
        logger.info(f"  策略: {strategy}")
        logger.info(f"  目标模块数: {len(target_modules) if isinstance(target_modules, list) else 'N/A'}")
        
        return lora_config
    
    def apply_lora(self, model: nn.Module) -> nn.Module:
        """
        应用LoRA到模型
        
        Args:
            model: 原始Diffold模型
        
        Returns:
            应用了LoRA的PEFT模型
        """
        if not self.enabled:
            logger.info("ℹ️ LoRA未启用，返回原始模型")
            return model
        
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT库未安装")
        
        logger.info("🔧 正在应用LoRA到模型...")
        
        # 创建LoRA配置（传入模型以获取安全的模块列表）
        lora_config = self.create_lora_config(model)
        
        # 应用LoRA
        try:
            peft_model = get_peft_model(model, lora_config)
            
            # 打印可训练参数统计
            trainable_params, all_params = self._count_parameters(peft_model)
            logger.info(f"✅ LoRA应用成功!")
            logger.info(f"📊 参数统计:")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  总参数: {all_params:,}")
            logger.info(f"  可训练比例: {100 * trainable_params / all_params:.2f}%")
            
            # 打印LoRA模块详情
            self._print_lora_modules(peft_model)
            
            return peft_model
            
        except Exception as e:
            logger.error(f"❌ LoRA应用失败: {e}")
            logger.info("💡 提示: 请检查target_modules是否正确匹配模型中的Linear层名称")
            raise
    
    def _count_parameters(self, model: nn.Module) -> tuple:
        """统计模型参数数量"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        return trainable_params, all_params
    
    def _print_lora_modules(self, peft_model: 'PeftModel'):
        """打印应用了LoRA的模块"""
        logger.info("📝 LoRA模块列表:")
        lora_modules = []
        for name, module in peft_model.named_modules():
            if 'lora' in name.lower():
                lora_modules.append(name)
        
        if lora_modules:
            # 只显示前10个和总数
            for i, name in enumerate(lora_modules[:10]):
                logger.info(f"  {i+1}. {name}")
            if len(lora_modules) > 10:
                logger.info(f"  ... 共 {len(lora_modules)} 个LoRA模块")
        else:
            logger.warning("⚠️ 未找到LoRA模块，可能target_modules配置不正确")
    
    @staticmethod
    def save_lora_weights(model: 'PeftModel', save_path: str):
        """
        保存LoRA权重（仅保存适配器权重，不保存基础模型）
        
        Args:
            model: PEFT模型
            save_path: 保存路径
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT库未安装")
        
        logger.info(f"💾 保存LoRA权重到: {save_path}")
        model.save_pretrained(save_path)
        logger.info("✅ LoRA权重保存成功")
    
    @staticmethod
    def load_lora_weights(base_model: nn.Module, lora_path: str) -> 'PeftModel':
        """
        加载LoRA权重到基础模型
        
        Args:
            base_model: 基础Diffold模型
            lora_path: LoRA权重路径
        
        Returns:
            加载了LoRA的PEFT模型
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT库未安装")
        
        logger.info(f"📥 从 {lora_path} 加载LoRA权重...")
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        logger.info("✅ LoRA权重加载成功")
        return peft_model
    
    @staticmethod
    def merge_and_unload(peft_model: 'PeftModel') -> nn.Module:
        """
        合并LoRA权重到基础模型并卸载适配器
        用于推理时获得完整权重的模型
        
        Args:
            peft_model: PEFT模型
        
        Returns:
            合并后的基础模型
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT库未安装")
        
        logger.info("🔄 合并LoRA权重到基础模型...")
        merged_model = peft_model.merge_and_unload()
        logger.info("✅ LoRA权重合并成功")
        return merged_model


def print_model_architecture(model: nn.Module, max_depth: int = 3):
    """
    打印模型架构，帮助确定LoRA目标模块名称
    
    Args:
        model: PyTorch模型
        max_depth: 最大打印深度
    """
    logger.info("🏗️ 模型架构分析:")
    
    def print_module(module, prefix='', depth=0):
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 检查是否包含Linear层
            has_linear = any(isinstance(m, nn.Linear) for m in child.modules())
            linear_mark = " 📍[含Linear层]" if has_linear else ""
            
            logger.info(f"{'  ' * depth}├─ {name}: {child.__class__.__name__}{linear_mark}")
            
            # 递归打印子模块
            if depth < max_depth:
                print_module(child, full_name, depth + 1)
    
    print_module(model)


def analyze_linear_layers(model: nn.Module) -> Dict[str, int]:
    """
    分析模型中的Linear层分布
    
    Args:
        model: PyTorch模型
    
    Returns:
        Linear层统计字典
    """
    stats = {
        'total_linear_layers': 0,
        'by_module': {}
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats['total_linear_layers'] += 1
            
            # 提取顶层模块名称
            top_module = name.split('.')[0] if '.' in name else name
            if top_module not in stats['by_module']:
                stats['by_module'][top_module] = 0
            stats['by_module'][top_module] += 1
    
    logger.info(f"🔍 Linear层分析:")
    logger.info(f"  总Linear层数: {stats['total_linear_layers']}")
    logger.info(f"  按模块分布:")
    for module_name, count in sorted(stats['by_module'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {module_name}: {count}层")
    
    return stats

