import os
import torch
import torch.nn as nn
from diffold.dataloader import RNA3DDataLoader
from diffold.diffold import Diffold
from rhofold.config import rhofold_config


def test_dataloader_creation():
    """测试DataLoader创建是否成功"""
    print("🧪 测试1: DataLoader创建...")
    
    try:
        data_loader = RNA3DDataLoader(
            batch_size=2,
            max_length=128,
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


def test_model_creation():
    """测试Diffold模型创建"""
    print("\n🧪 测试3: Diffold模型创建...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                print(f"  {key}: 列表长度 {len(value)}")
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


def test_training_compatibility():
    """测试训练兼容性"""
    print("\n🧪 测试5: 训练兼容性...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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


if __name__ == "__main__":
    print("🧪 RNA3D DataLoader 与 Diffold 模型兼容性测试")
    print("=" * 60)
    
    # 测试步骤
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
            success3, model, device = test_model_creation()
            results.append(("模型创建", success3))
            
            if success3 and batch is not None:
                # 测试4: 前向传播
                success4 = test_forward_pass(model, batch, device)
                results.append(("前向传播", success4))
    
    # 测试5: 训练兼容性（独立测试）
    success5 = test_training_compatibility()
    results.append(("训练兼容性", success5))
    
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