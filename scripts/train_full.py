#!/usr/bin/env python3
"""
TrajectoryForge FULL Training - All Difficulties, Maximum Compute

This script trains on ALL difficulty levels with 10x more data and compute.
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# For data generation we still use JAX on CPU
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS


# =============================================================================
# PyTorch Model - LARGER Architecture
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """Larger Transformer encoder for trajectories."""
    
    def __init__(self, hidden_dim=512, latent_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(self, x):
        # x: (batch, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_proj(x)
        return x


class ForceFieldDecoder(nn.Module):
    """Larger decoder for wind field parameters."""
    
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Wind field outputs
        self.wind_center = nn.Linear(hidden_dim, 2)
        self.wind_size = nn.Linear(hidden_dim, 2)
        self.wind_direction = nn.Linear(hidden_dim, 2)
        self.wind_strength = nn.Linear(hidden_dim, 1)
        
    def forward(self, z):
        h = self.net(z)
        
        center = self.wind_center(h)
        size = torch.nn.functional.softplus(self.wind_size(h)) + 0.5
        direction = torch.tanh(self.wind_direction(h))
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        strength = torch.nn.functional.softplus(self.wind_strength(h)) * 10  # Scale strength
        
        return {
            'center': center,
            'size': size,
            'direction': direction,
            'strength': strength.squeeze(-1)
        }


class InverseModel(nn.Module):
    """Full model: trajectory -> force field."""
    
    def __init__(self, hidden_dim=512, latent_dim=256):
        super().__init__()
        self.encoder = TrajectoryEncoder(hidden_dim, latent_dim)
        self.decoder = ForceFieldDecoder(latent_dim, hidden_dim)
        
    def forward(self, trajectory):
        z = self.encoder(trajectory)
        params = self.decoder(z)
        return params


# =============================================================================
# Dataset with variable length support
# =============================================================================

class TrajectoryDataset(Dataset):
    """Dataset of trajectories and their generating force fields."""
    
    def __init__(self, samples, target_length=100):
        self.samples = samples
        self.target_length = target_length
        
    def __len__(self):
        return len(self.samples)
    
    def _resample_trajectory(self, traj, target_len):
        """Resample trajectory to target length using linear interpolation."""
        current_len = len(traj)
        if current_len == target_len:
            return traj
        
        # Create interpolation indices
        old_indices = np.linspace(0, current_len - 1, current_len)
        new_indices = np.linspace(0, current_len - 1, target_len)
        
        # Interpolate each dimension
        new_traj = np.zeros((target_len, 2))
        new_traj[:, 0] = np.interp(new_indices, old_indices, traj[:, 0])
        new_traj[:, 1] = np.interp(new_indices, old_indices, traj[:, 1])
        
        return new_traj
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        traj = np.array(sample['trajectory'])
        
        # Resample to target length
        traj = self._resample_trajectory(traj, self.target_length)
        
        return {
            'trajectory': torch.tensor(traj, dtype=torch.float32),
            'initial_position': torch.tensor(np.array(sample['initial_position']), dtype=torch.float32),
            'initial_velocity': torch.tensor(np.array(sample['initial_velocity']), dtype=torch.float32),
            'difficulty': sample.get('difficulty', 'unknown'),
        }


# =============================================================================
# Physics Simulation (PyTorch - differentiable)
# =============================================================================

def simulate_trajectory_torch(init_pos, init_vel, wind_center, wind_size, wind_direction, wind_strength, 
                               dt=0.01, num_steps=100, device='cuda'):
    """Simulate trajectory using PyTorch (differentiable)."""
    
    position = init_pos.clone()
    velocity = init_vel.clone()
    
    positions = [position.clone()]
    
    softness = 0.1
    
    for _ in range(num_steps - 1):
        # Compute wind force
        rel_pos = (position - wind_center) / wind_size
        scale = 1.0 / softness
        indicators = torch.sigmoid(scale * (1.0 - torch.abs(rel_pos)))
        indicator = indicators.prod(dim=-1, keepdim=True)
        
        force = indicator * wind_direction * wind_strength.unsqueeze(-1)
        
        # Semi-implicit Euler
        velocity = velocity + force * dt
        position = position + velocity * dt
        
        positions.append(position.clone())
    
    return torch.stack(positions, dim=1)


def trajectory_loss(predicted_traj, target_traj):
    """MSE loss between trajectories."""
    return ((predicted_traj - target_traj) ** 2).mean()


def endpoint_loss(predicted_traj, target_traj):
    """Extra penalty for endpoint mismatch."""
    return ((predicted_traj[:, -1, :] - target_traj[:, -1, :]) ** 2).mean()


# =============================================================================
# Training with curriculum
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, dt=0.01, num_steps=100, 
                endpoint_weight=0.5):
    """Train for one epoch with combined losses."""
    model.train()
    total_loss = 0
    total_traj_loss = 0
    total_endpoint_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        trajectory = batch['trajectory'].to(device)
        init_pos = batch['initial_position'].to(device)
        init_vel = batch['initial_velocity'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        params = model(trajectory)
        
        # Simulate with predicted parameters
        pred_traj = simulate_trajectory_torch(
            init_pos, init_vel,
            params['center'], params['size'], 
            params['direction'], params['strength'],
            dt=dt, num_steps=num_steps, device=device
        )
        
        # Compute losses
        traj_loss = trajectory_loss(pred_traj, trajectory)
        end_loss = endpoint_loss(pred_traj, trajectory)
        loss = traj_loss + endpoint_weight * end_loss
        
        # Skip NaN losses
        if torch.isnan(loss):
            continue
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        total_traj_loss += traj_loss.item()
        total_endpoint_loss += end_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'traj_loss': total_traj_loss / num_batches,
        'endpoint_loss': total_endpoint_loss / num_batches,
    }


def evaluate(model, dataloader, device, dt=0.01, num_steps=100, endpoint_weight=0.5):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_traj_loss = 0
    total_endpoint_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            trajectory = batch['trajectory'].to(device)
            init_pos = batch['initial_position'].to(device)
            init_vel = batch['initial_velocity'].to(device)
            
            params = model(trajectory)
            
            pred_traj = simulate_trajectory_torch(
                init_pos, init_vel,
                params['center'], params['size'], 
                params['direction'], params['strength'],
                dt=dt, num_steps=num_steps, device=device
            )
            
            traj_loss = trajectory_loss(pred_traj, trajectory)
            end_loss = endpoint_loss(pred_traj, trajectory)
            loss = traj_loss + endpoint_weight * end_loss
            
            total_loss += loss.item()
            total_traj_loss += traj_loss.item()
            total_endpoint_loss += end_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'traj_loss': total_traj_loss / num_batches,
        'endpoint_loss': total_endpoint_loss / num_batches,
    }


# =============================================================================
# Data Generation - ALL DIFFICULTIES
# =============================================================================

def generate_training_data_all_difficulties(num_samples_per_difficulty, seed=42):
    """Generate training data from ALL difficulty levels."""
    print(f"Generating {num_samples_per_difficulty} samples per difficulty level...")
    
    key = random.PRNGKey(seed)
    all_samples = []
    
    difficulties = ['easy', 'medium', 'hard', 'expert']
    
    for diff_name in difficulties:
        level = DIFFICULTY_LEVELS[diff_name]
        print(f"  Generating {diff_name} samples...")
        
        samples = []
        for i in tqdm(range(num_samples_per_difficulty), desc=f"  {diff_name}", leave=False):
            key, subkey = random.split(key)
            try:
                sample = generate_sample_at_difficulty(subkey, level)
                sample['difficulty'] = diff_name
                samples.append(sample)
            except Exception as e:
                # Skip failed samples
                continue
        
        print(f"    Generated {len(samples)} {diff_name} samples")
        all_samples.extend(samples)
    
    # Shuffle all samples
    np.random.seed(seed)
    np.random.shuffle(all_samples)
    
    print(f"Total samples: {len(all_samples)}")
    return all_samples


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train TrajectoryForge - FULL')
    parser.add_argument('--samples-per-difficulty', type=int, default=12500, 
                        help='Samples per difficulty level (total = 4x this)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--output', type=str, default='models/trained_model_full.pt', 
                        help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='LR warmup epochs')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("TRAJECTORYFORGE FULL TRAINING - ALL DIFFICULTIES")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Generate data from ALL difficulties
    total_samples = args.samples_per_difficulty * 4
    print(f"\n📦 Generating {total_samples} samples ({args.samples_per_difficulty} per difficulty)...")
    samples = generate_training_data_all_difficulties(args.samples_per_difficulty, args.seed)
    
    # Split into train/val
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    
    print(f"\n📦 Data Split:")
    print(f"   Train: {len(train_samples)}")
    print(f"   Val: {len(val_samples)}")
    
    # Count by difficulty
    for diff in ['easy', 'medium', 'hard', 'expert']:
        train_count = sum(1 for s in train_samples if s.get('difficulty') == diff)
        val_count = sum(1 for s in val_samples if s.get('difficulty') == diff)
        print(f"   {diff}: {train_count} train, {val_count} val")
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_samples, target_length=100)
    val_dataset = TrajectoryDataset(val_samples, target_length=100)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # Create model
    model = InverseModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 Model: {n_params:,} parameters")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Latent dim: {args.latent_dim}")
    
    # Optimizer with warmup + cosine decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Warmup + Cosine Annealing scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Warmup epochs: {args.warmup_epochs}")
    print("=" * 70)
    
    history = {'train_loss': [], 'val_loss': [], 'train_traj': [], 'train_end': [],
               'val_traj': [], 'val_end': [], 'lr': []}
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_traj'].append(train_metrics['traj_loss'])
        history['train_end'].append(train_metrics['endpoint_loss'])
        history['val_traj'].append(val_metrics['traj_loss'])
        history['val_end'].append(val_metrics['endpoint_loss'])
        history['lr'].append(current_lr)
        
        # Save best model (or if val is NaN, save based on train loss)
        save_model = False
        if not np.isnan(val_metrics['loss']) and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_model = True
        elif np.isnan(val_metrics['loss']) and train_metrics['loss'] < best_val_loss:
            best_val_loss = train_metrics['loss']
            save_model = True
        
        if save_model:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'args': vars(args),
                'history': history,
            }, args.output)
        
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{args.epochs}: "
                  f"train={train_metrics['loss']:.4f} (traj={train_metrics['traj_loss']:.4f}, end={train_metrics['endpoint_loss']:.4f}), "
                  f"val={val_metrics['loss']:.4f}, best={best_val_loss:.4f}, "
                  f"lr={current_lr:.2e}, time={elapsed/60:.1f}min")
    
    elapsed = time.time() - start_time
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Model saved to: {args.output}")
    print("=" * 70)
    
    # Always save final model
    final_output = args.output.replace('.pt', '_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'val_loss': history['val_loss'][-1],
        'train_loss': history['train_loss'][-1],
        'args': vars(args),
        'history': history,
    }, final_output)
    print(f"   Final model also saved to: {final_output}")


if __name__ == '__main__':
    main()

