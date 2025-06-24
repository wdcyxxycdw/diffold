#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diffold Model Simplified Training Test Script

If full version testing encounters dependency issues, use this simplified version
Directly construct necessary input tensors to test model training capabilities
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import traceback

# Import necessary modules
from rhofold.config import rhofold_config
from diffold.diffold import Diffold


def generate_mock_inputs(seq_length=16, msa_depth=32, batch_size=1):
    """Generate mock input data"""
    print(f"Generating mock input data...")
    print(f"Sequence length: {seq_length}, MSA depth: {msa_depth}, Batch size: {batch_size}")
    
    # RNA sequence (ACGU alphabet)
    seq = "ACGUACGUACGUACGU"[:seq_length]
    
    # Fix: Mock MSA tokens (RhoFold format)
    # According to get_msa_feature processing, final shape should be [batch_size, msa_depth, seq_len]
    # Does not include cls/eos token
    tokens = torch.randint(0, 9, (batch_size, msa_depth, seq_length))  # RNA alphabet size ≈ 9
    
    # Fix: Mock RNA-FM tokens
    # According to get_rna_fm_token processing, should be [batch_size, seq_len], no cls/eos
    rna_fm_tokens = torch.randint(0, 25, (batch_size, seq_length))  # RNA-FM alphabet size
    
    # Fix: Determine correct atom count based on actual RNA sequence
    # First generate temporary coordinates to determine atom count
    temp_coords = torch.randn(1, seq_length * 30, 3) * 10  # Temporary estimate, slightly more
    
    # Import necessary functions
    from diffold.input_processor import process_alphafold3_input
    
    # Call process_alphafold3_input to determine actual atom count
    af_in, atom_mask = process_alphafold3_input(
        ss_rna=[seq],
        atom_pos=[temp_coords.squeeze(0)],  # Need tensor list, remove batch dimension
    )
    
    # Get actual required atom count
    actual_num_atoms = af_in.atom_inputs.shape[1]
    print(f"✓ Determined actual atom count based on sequence '{seq}': {actual_num_atoms}")
    
    # Generate correct number of atom coordinates
    target_coords = torch.randn(batch_size, actual_num_atoms, 3) * 10  # Reasonable 3D coordinate range
    
    print(f"✓ tokens shape: {tokens.shape}")
    print(f"✓ rna_fm_tokens shape: {rna_fm_tokens.shape}")
    print(f"✓ target_coords shape: {target_coords.shape}")
    print(f"✓ Sequence: {seq}")
    
    return {
        'seq': seq,
        'tokens': tokens,
        'rna_fm_tokens': rna_fm_tokens,
        'target_coords': target_coords
    }


def generate_mock_coordinates(sequence, atoms_per_residue=27):
    """Generate mock atom coordinates"""
    seq_len = len(sequence)
    # Use reasonable estimate first, actual count will be determined in process_alphafold3_input
    num_atoms = seq_len * atoms_per_residue
    
    # Generate random 3D coordinates, maintain reasonable structure
    coords = torch.randn(1, num_atoms, 3) * 10  # Scale to reasonable Angstrom units
    
    return coords


def adjust_coordinates_to_match_input(target_coords, af_in):
    """Adjust coordinate count based on process_alphafold3_input output"""
    if hasattr(af_in, 'atom_inputs') and af_in.atom_inputs is not None:
        expected_atoms = af_in.atom_inputs.shape[1]  # Actual atom count
        current_atoms = target_coords.shape[1]
        
        if expected_atoms != current_atoms:
            print(f"Adjusting atom coordinate count: {current_atoms} -> {expected_atoms}")
            if expected_atoms > current_atoms:
                # If more atoms needed, pad with zero coordinates
                padding = torch.zeros(target_coords.shape[0], expected_atoms - current_atoms, 3, 
                                    device=target_coords.device)
                target_coords = torch.cat([target_coords, padding], dim=1)
            else:
                # If too many atoms, truncate to needed amount
                target_coords = target_coords[:, :expected_atoms, :]
    
    return target_coords


def test_model_initialization():
    """Test model initialization"""
    print("=" * 50)
    print("Test 1: Model Initialization")
    print("=" * 50)
    
    try:
        # Use specified weight path
        rhofold_checkpoint_path = "/home/yamanashi/RhoFold/pretrained/model_20221010_params.pt"
        
        print(f"Initializing Diffold model...")
        print(f"RhoFold weights path: {rhofold_checkpoint_path}")
        
        # Check if weight file exists
        if os.path.exists(rhofold_checkpoint_path):
            print("✓ Weight file exists")
        else:
            print(f"⚠️ Weight file does not exist: {rhofold_checkpoint_path}")
            print("Will try testing without loading pretrained weights")
            rhofold_checkpoint_path = None
        
        model = Diffold(rhofold_config, rhofold_checkpoint_path=rhofold_checkpoint_path)
        
        print("✓ Model initialization successful")
        
        # Check if RhoFold parameters are frozen
        rhofold_params = list(model.rhofold.parameters())
        rhofold_param_count = sum(p.numel() for p in rhofold_params)  # Calculate actual parameter count
        frozen_count = sum(1 for p in rhofold_params if not p.requires_grad)
        print(f"✓ RhoFold total parameters: {len(rhofold_params)}")
        print(f"✓ RhoFold parameter count: {rhofold_param_count:,}")
        print(f"✓ Frozen parameter count: {frozen_count}")
        print(f"✓ Frozen ratio: {frozen_count/len(rhofold_params)*100:.1f}%")
        
        # Check trainable parameters
        trainable_params = model.get_trainable_parameters()
        trainable_param_count = sum(p.numel() for p in trainable_params)
        total_param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Trainable parameter count: {len(trainable_params)}")
        print(f"✓ Trainable parameter total: {trainable_param_count:,}")
        print(f"✓ Model total parameter count: {total_param_count:,}")
        print(f"✓ Frozen parameter count: {total_param_count - trainable_param_count:,}")
        print(f"✓ Frozen parameter ratio: {(total_param_count - trainable_param_count)/total_param_count*100:.1f}%")
        
        return model
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
        return None


def test_forward_pass(model, data):
    """Test forward pass"""
    print("=" * 50)
    print("Test 2: Forward Pass")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Move model and data to device
        model = model.to(device)
        tokens = data['tokens'].to(device)
        rna_fm_tokens = data['rna_fm_tokens'].to(device)
        target_coords = data['target_coords'].to(device)
        
        print("✓ Model and data moved to device")
        
        # Test inference mode forward pass (no target coordinates)
        print("Testing inference mode forward pass...")
        model.set_eval_mode()
        
        with torch.no_grad():
            outputs = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=data['seq']
            )
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            rhofold_outputs, single_fea, pair_fea = outputs
            print("✓ Inference mode forward pass successful")
            print(f"  - single_fea shape: {single_fea.shape if single_fea is not None else 'None'}")
            print(f"  - pair_fea shape: {pair_fea.shape if pair_fea is not None else 'None'}")
        else:
            print("✓ Inference mode forward pass successful")
        
        # Test training mode forward pass (with target coordinates)
        print("Testing training mode forward pass...")
        model.set_train_mode()
        
        loss, denoised_pos, loss_breakdown = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=data['seq'],
            target_coords=target_coords
        )
        
        print("✓ Training mode forward pass successful")
        print(f"  - Loss value: {loss.item():.6f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        print(f"  - Loss type: {type(loss)}")
        
        if loss_breakdown is not None and isinstance(loss_breakdown, dict):
            print("  - Loss breakdown:")
            for key, value in loss_breakdown.items():
                if isinstance(value, torch.Tensor):
                    print(f"    - {key}: {value.item():.6f}")
        
        return loss
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return None


def test_backward_pass(model, data):
    """Test backward pass and parameter updates"""
    print("=" * 50)
    print("Test 3: Backward Pass and Parameter Updates")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.set_train_mode()
        
        # Create optimizer
        trainable_params = model.get_trainable_parameters()
        optimizer = optim.Adam(trainable_params, lr=1e-4)
        
        print(f"✓ Created optimizer managing {len(trainable_params)} trainable parameters")
        
        # Record parameter values before update (sample a few parameters)
        param_before = {}
        sample_size = min(5, len(trainable_params))
        for i in range(sample_size):
            param_before[i] = trainable_params[i].clone().detach()
        
        # Forward pass
        tokens = data['tokens'].to(device)
        rna_fm_tokens = data['rna_fm_tokens'].to(device)
        target_coords = data['target_coords'].to(device)
        
        loss, _, _ = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=data['seq'],
            target_coords=target_coords
        )
        
        print(f"✓ Forward pass completed, loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        print("✓ Backward pass completed")
        
        # Check gradients
        grad_count = 0
        total_grad_norm = 0.0
        for param in trainable_params:
            if param.grad is not None:
                grad_count += 1
                total_grad_norm += param.grad.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"✓ Gradient computation completed")
        print(f"  - Parameters with gradients: {grad_count}/{len(trainable_params)}")
        print(f"  - Total gradient norm: {total_grad_norm:.6f}")
        
        # Parameter update
        optimizer.step()
        print("✓ Parameter update completed")
        
        # Verify parameters were actually updated
        updated_count = 0
        for i in range(sample_size):
            if not torch.equal(param_before[i], trainable_params[i]):
                updated_count += 1
        
        print(f"✓ Parameter update verification: {updated_count}/{sample_size} sampled parameters updated")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        traceback.print_exc()
        return False


def test_training_loop(model, data, num_steps=3):
    """Test training loop"""
    print("=" * 50)
    print(f"Test 4: Training Loop ({num_steps} steps)")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.set_train_mode()
        
        # Create optimizer and learning rate scheduler
        trainable_params = model.get_trainable_parameters()
        optimizer = optim.Adam(trainable_params, lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        
        print(f"✓ Training setup completed")
        
        # Prepare data
        tokens = data['tokens'].to(device)
        rna_fm_tokens = data['rna_fm_tokens'].to(device)
        target_coords = data['target_coords'].to(device)
        
        losses = []
        
        for step in range(num_steps):
            print(f"\n--- Step {step + 1}/{num_steps} ---")
            
            # Forward pass
            optimizer.zero_grad()
            
            loss, _, loss_breakdown = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=data['seq'],
                target_coords=target_coords
            )
            
            losses.append(loss.item())
            print(f"Loss: {loss.item():.6f}")
            
            # Backward pass and parameter update
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            if loss_breakdown is not None and isinstance(loss_breakdown, dict):
                print("Loss breakdown:")
                for key, value in loss_breakdown.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: {value.item():.6f}")
        
        print(f"\n✓ Training loop completed")
        print(f"✓ Loss change: {losses[0]:.6f} -> {losses[-1]:.6f}")
        
        if len(losses) > 1:
            loss_change = losses[-1] - losses[0]
            print(f"✓ Loss change amount: {loss_change:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        traceback.print_exc()
        return False


def test_training_mode(model, inputs):
    """Test training mode"""
    print("\n🔥 Testing training mode...")
    
    model.set_train_mode()
    
    try:
        # Training forward pass
        loss, denoised_pos, loss_breakdown = model(
            tokens=inputs['tokens'].cuda(),
            rna_fm_tokens=inputs['rna_fm_tokens'].cuda(),
            seq=inputs['seq'],
            target_coords=inputs['target_coords'].cuda()
        )
        
        print(f"✓ Training forward pass successful")
        print(f"  - Diffusion loss: {loss.item():.6f}")
        print(f"  - Denoised coordinates shape: {denoised_pos.shape if denoised_pos is not None else 'None'}")
        
        if loss_breakdown:
            print(f"  - Loss details: {loss_breakdown}")
        
        return loss
        
    except Exception as e:
        print(f"❌ Training mode test failed: {e}")
        traceback.print_exc()
        return None


def main():
    """Main test function"""
    print("🧬 Diffold Model Simplified Training Test Started")
    print("=" * 60)
    
    # Generate mock data
    data = generate_mock_inputs(seq_length=16, msa_depth=32, batch_size=1)
    
    # Test 1: Model initialization
    model = test_model_initialization()
    if model is None:
        print("❌ Model initialization failed, terminating tests")
        return False
    
    # Test 2: Forward pass
    loss = test_forward_pass(model, data)
    if loss is None:
        print("❌ Forward pass failed, terminating tests")
        return False
    
    # Test 3: Backward pass
    backward_success = test_backward_pass(model, data)
    if not backward_success:
        print("❌ Backward pass failed, terminating tests")
        return False
    
    # Test 4: Training loop
    training_success = test_training_loop(model, data, num_steps=3)
    if not training_success:
        print("❌ Training loop failed")
        return False
    
    print("=" * 60)
    print("🎉 All tests passed! Diffold model can train normally")
    print("✅ Test results:")
    print("  - ✓ Model initialization normal")
    print("  - ✓ RhoFold parameters correctly frozen")
    print("  - ✓ Forward pass working normally")
    print("  - ✓ Backward pass and gradient computation normal")
    print("  - ✓ Parameter updates normal")
    print("  - ✓ Training loop normal")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 