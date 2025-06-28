import os
import torch
import torch.nn as nn
from diffold.dataloader import RNA3DDataLoader, RNA3DDataset
from diffold.diffold import Diffold
from rhofold.config import rhofold_config


def test_dataloader_creation():
    """测试DataLoader创建是否成功"""
    print("🧪 测试1: DataLoader创建...")
    
    try:
        data_loader = RNA3DDataLoader(
            batch_size=1,
            max_length=16,
            num_workers=0,
            force_reload=False
        )
        print("✅ DataLoader创建成功")
        return True, data_loader
    except Exception as e:
        print(f"❌ DataLoader创建失败: {e}")
        return False, None


def test_data_loading(data_loader):
    """测试数据加载"""
    print("\n🧪 测试2: 数据加载...")
    
    try:
        train_loader = data_loader.get_train_dataloader(fold=0)
        
        # 获取数据集大小
        dataset_size = 0
        try:
            dataset_size = len(train_loader.dataset)
        except:
            # 如果len失败，尝试迭代统计
            for i, _ in enumerate(train_loader):
                dataset_size = i + 1
                if i >= 10:  # 只统计前10个避免过长
                    break
        
        print(f"✅ 训练数据加载器创建成功，数据集大小: {dataset_size}")
        
        if dataset_size == 0:
            print("⚠️ 警告: 训练数据集为空")
            return False, None
        
        # 尝试获取一个批次
        batch = next(iter(train_loader))
        print("✅ 成功获取一个批次的数据")
        return True, batch
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation(device):
    """测试Diffold模型创建"""
    print("\n🧪 测试3: Diffold模型创建...")
    
    try:
        print(f"📱 使用设备: {device}")
        
        config = rhofold_config
        model = Diffold(config)
        model = model.to(device)
        model.set_train_mode()
        
        print("✅ Diffold模型创建成功")
        return True, model, device
        
    except Exception as e:
        print(f"❌ Diffold模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_forward_pass(model, batch, device):
    """测试模型前向传播"""
    print("\n🧪 测试4: 模型前向传播...")
    
    try:
        # 打印批次信息
        print("📊 批次数据信息:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {value}, {len(value[0])}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 准备输入数据
        tokens = batch['tokens'].to(device)
        rna_fm_tokens = batch['rna_fm_tokens'].to(device)
        seq = batch['sequences'][0] if isinstance(batch['sequences'], list) else batch['sequences']
        missing_atom_mask = batch['missing_atom_masks'].to(device)
        target_coords = batch.get('coordinates', None)
        if target_coords is not None:
            target_coords = target_coords.to(device)
        
        print("🚀 开始模型前向传播...")
        
        with torch.no_grad():
            output = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=seq,
                target_coords=target_coords,
                missing_atom_mask=missing_atom_mask
            )
        
        print("✅ 模型前向传播成功!")
        
        # 打印输出信息
        if isinstance(output, tuple):
            print(f"📤 输出类型: tuple, 长度: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  输出 {i}: {out.shape}")
                elif out is None:
                    print(f"  输出 {i}: None")
                else:
                    print(f"  输出 {i}: {type(out)}")
        else:
            if isinstance(output, torch.Tensor):
                print(f"📤 输出形状: {output.shape}")
            else:
                print(f"📤 输出类型: {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_compatibility(device):
    """测试训练兼容性"""
    print("\n🧪 测试5: 训练兼容性...")
    
    try:
        # 创建简化的数据加载器
        data_loader = RNA3DDataLoader(
            batch_size=1,
            max_length=64,  # 使用更短的序列
            num_workers=0,
            force_reload=False
        )
        
        train_loader = data_loader.get_train_dataloader(fold=0)
        
        # 创建模型
        config = rhofold_config
        model = Diffold(config).to(device)
        model.set_train_mode()
        
        # 获取可训练参数
        trainable_params = model.get_trainable_parameters()
        param_count = sum(p.numel() for p in trainable_params)
        print(f"📈 可训练参数数量: {param_count:,}")
        
        # 创建优化器
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        
        # 执行一步训练
        batch = next(iter(train_loader))
        
        tokens = batch['tokens'].to(device)
        rna_fm_tokens = batch['rna_fm_tokens'].to(device)
        seq = batch['sequences'][0] if isinstance(batch['sequences'], list) else batch['sequences']
        seq = seq
        target_coords = batch.get('coordinates', None).to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        result = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=seq,
            target_coords=target_coords
        )
        
        # 处理损失
        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result
        
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            print(f"✅ 损失计算成功: {loss.item():.6f}")
            
            # 反向传播
            loss.backward()
            optimizer.step()
            print("✅ 反向传播成功!")
            return True
        else:
            print(f"⚠️ 损失不支持梯度: {loss}")
            return False
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_samples(data_dir="/home/wdcyx/rhofold/RNA3D_DATA", fold=0, split="train", max_count=20):
    """
    列出可用的样本名称
    
    Args:
        data_dir: 数据目录
        fold: 交叉验证折数
        split: 数据集分割
        max_count: 最大显示数量
    """
    print(f"📋 列出 fold {fold} {split} 数据集中的可用样本...")
    
    try:
        dataset = RNA3DDataset(
            data_dir=data_dir,
            fold=fold,
            split=split,
            max_length=512,
            use_msa=True,
            force_reload=False
        )
        
        sample_names = [sample['name'] for sample in dataset.samples]
        print(f"✅ 找到 {len(sample_names)} 个样本")
        
        # 显示前max_count个样本
        display_names = sample_names[:max_count] if max_count else sample_names
        for i, name in enumerate(display_names):
            print(f"  {i+1:3d}. {name}")
        
        if len(sample_names) > max_count:
            print(f"  ... 还有 {len(sample_names) - max_count} 个样本")
        
        return sample_names
        
    except Exception as e:
        print(f"❌ 列出样本失败: {e}")
        return []


def get_specific_sample_batch(sample_name, data_dir="/home/wdcyx/rhofold/RNA3D_DATA", fold=0, split="train"):
    """
    获取特定样本的批次数据
    
    Args:
        sample_name: 样本名称
        data_dir: 数据目录
        fold: 交叉验证折数
        split: 数据集分割
    
    Returns:
        批次数据字典，如果未找到则返回None
    """
    print(f"🔍 正在获取样本: {sample_name}")
    
    try:
        dataset = RNA3DDataset(
            data_dir=data_dir,
            fold=fold,
            split=split,
            max_length=512,
            use_msa=True,
            force_reload=False
        )
        
        # 查找匹配的样本
        target_idx = None
        for idx, sample in enumerate(dataset.samples):
            if sample['name'] == sample_name:
                target_idx = idx
                break
        
        if target_idx is None:
            print(f"❌ 未找到样本: {sample_name}")
            available_names = [s['name'] for s in dataset.samples[:10]]
            print(f"   前10个可用样本: {available_names}")
            return None
        
        # 获取样本数据
        sample_data = dataset[target_idx]
        print(f"✅ 成功找到样本: {sample_name}")
        
        # 类型检查和处理
        if not isinstance(sample_data, dict):
            print(f"❌ 样本数据格式错误: {type(sample_data)}")
            return None
            
        sequence = sample_data.get('sequence', '')
        if isinstance(sequence, str):
            print(f"   序列长度: {len(sequence)}")
            print(f"   序列: {sequence}")
        
        # 确保必要的字段存在且类型正确
        tokens = sample_data.get('tokens')
        rna_fm_tokens = sample_data.get('rna_fm_tokens')
        coordinates = sample_data.get('coordinates')
        seq_length = sample_data.get('seq_length', 0)
        
        if not isinstance(tokens, torch.Tensor):
            print(f"❌ tokens类型错误: {type(tokens)}")
            return None
            
        if not isinstance(coordinates, torch.Tensor):
            print(f"❌ coordinates类型错误: {type(coordinates)}")
            return None
        
        # 转换为批次格式（batch_size=1）
        batch = {
            'names': [sample_data.get('name', sample_name)],
            'sequences': [sequence],
            'tokens': tokens.unsqueeze(0),  # 添加batch维度
            'rna_fm_tokens': rna_fm_tokens.unsqueeze(0) if isinstance(rna_fm_tokens, torch.Tensor) else None,
            'coordinates': coordinates.unsqueeze(0),  # 添加batch维度
            'seq_lengths': torch.tensor([seq_length], dtype=torch.long)
        }
        
        # 处理缺失原子掩码（如果存在）
        missing_atom_mask = sample_data.get('missing_atom_mask')
        if isinstance(missing_atom_mask, torch.Tensor):
            batch['missing_atom_masks'] = missing_atom_mask.unsqueeze(0)
            batch['atom_masks'] = torch.ones(1, coordinates.shape[0], dtype=torch.bool)
            num_atoms = sample_data.get('num_atoms', coordinates.shape[0])
            batch['num_atoms'] = torch.tensor([num_atoms], dtype=torch.long)
        
        return batch
        
    except Exception as e:
        print(f"❌ 获取样本失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_tests_with_specific_sample(sample_name, fold=0, split="train", device=None):
    """
    使用特定样本运行测试
    
    Args:
        sample_name: 样本名称
        fold: 交叉验证折数
        split: 数据集分割
        device: 设备
    """
    print(f"🧪 使用特定样本进行测试: {sample_name}")
    print("=" * 60)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    # 获取特定样本的批次数据
    print("🎯 获取特定样本数据...")
    batch = get_specific_sample_batch(sample_name, fold=fold, split=split)
    
    if batch is None:
        print("❌ 无法获取样本数据，测试终止")
        return False
    
    results.append(("样本数据获取", True))
    
    # 测试3: 模型创建
    success3, model, device = test_model_creation(device)
    results.append(("模型创建", success3))
    
    if success3 and batch is not None:
        # 测试4: 前向传播
        success4 = test_forward_pass(model, batch, device)
        results.append(("前向传播", success4))
        
        # 可选：测试训练兼容性（使用这个特定样本）
        if model is not None:
            print("\n🧪 测试训练兼容性（使用特定样本）...")
            try:
                model.train()  # 设置为训练模式
                
                # 获取可训练参数
                trainable_params = model.get_trainable_parameters()
                param_count = sum(p.numel() for p in trainable_params)
                print(f"📈 可训练参数数量: {param_count:,}")
                
                # 创建优化器
                optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
                
                # 准备输入数据
                tokens = batch['tokens'].to(device)
                rna_fm_tokens = batch['rna_fm_tokens'].to(device) if batch['rna_fm_tokens'] is not None else None
                seq = batch['sequences'][0]
                target_coords = batch['coordinates'].to(device)
                missing_atom_mask = batch.get('missing_atom_masks')
                if missing_atom_mask is not None:
                    missing_atom_mask = missing_atom_mask.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                result = model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=seq,
                    target_coords=target_coords,
                    missing_atom_mask=missing_atom_mask
                )
                
                # 处理损失
                if isinstance(result, tuple):
                    loss = result[0]
                else:
                    loss = result
                
                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    print(f"✅ 损失计算成功: {loss.item():.6f}")
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    print("✅ 反向传播成功!")
                    results.append(("训练兼容性", True))
                else:
                    print(f"⚠️ 损失不支持梯度: {loss}")
                    results.append(("训练兼容性", False))
                    
            except Exception as e:
                print(f"❌ 训练兼容性测试失败: {e}")
                import traceback
                traceback.print_exc()
                results.append(("训练兼容性", False))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        all_passed = all_passed and success
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RNA3D DataLoader 与 Diffold 模型测试")
    parser.add_argument("--sample", type=str, help="指定要测试的样本名称，如 '3meiA'", default="5on3_E")
    parser.add_argument("--list", action="store_true", help="列出所有可用样本")
    parser.add_argument("--fold", type=int, default=0, help="交叉验证折数 (0-9)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid"], help="数据集分割")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="设备选择")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🧪 RNA3D DataLoader 与 Diffold 模型兼容性测试")
    print("=" * 60)
    
    if args.list:
        # 列出可用样本
        list_available_samples(fold=args.fold, split=args.split)
        
    elif args.sample:
        # 测试特定样本
        success = run_tests_with_specific_sample(args.sample, args.fold, args.split, device)
        if success:
            print("\n🎉 特定样本测试成功完成!")
        else:
            print("\n❌ 特定样本测试失败!")
            exit(1)
            
    else:
        # 原始的随机测试流程
        print("🔀 使用随机样本进行测试...")
        
        results = []
        
        # 测试1: DataLoader创建
        success1, data_loader = test_dataloader_creation()
        results.append(("DataLoader创建", success1))
        
        if not success1:
            print("\n❌ DataLoader创建失败，跳过后续测试")
        else:
            # 测试2: 数据加载
            success2, batch = test_data_loading(data_loader)
            results.append(("数据加载", success2))
            
            if not success2:
                print("\n❌ 数据加载失败，跳过需要数据的测试")
            else:
                # 测试3: 模型创建
                success3, model, device = test_model_creation(device)
                results.append(("模型创建", success3))
                
                if success3 and batch is not None:
                    # 测试4: 前向传播
                    success4 = test_forward_pass(model, batch, device)
                    results.append(("前向传播", success4))
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print("📊 测试结果总结:")
        
        all_passed = True
        for test_name, success in results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {test_name}: {status}")
            all_passed = all_passed and success
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 所有测试通过! DataLoader可以与Diffold模型配合使用进行训练")
        else:
            print("⚠️ 部分测试失败，建议检查:")
            print("  1. 数据目录是否存在且有数据")
            print("  2. 相关依赖是否正确安装")
            print("  3. 模型配置是否正确")
            
    print("\n💡 使用提示:")
    print("  python test_dataloader.py --list                    # 列出可用样本")
    print("  python test_dataloader.py --sample 3meiA           # 测试特定样本")
    print("  python test_dataloader.py --sample 3owzA --fold 1  # 测试fold 1中的样本")
    print("  python test_dataloader.py                          # 随机测试")