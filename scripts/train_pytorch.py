#!/usr/bin/env python3
"""
TrajectoryForge PyTorch Training Script

Uses PyTorch for GPU training since JAX doesn't support CUDA 12.9 yet.
This is a simplified training loop that trains a model to predict force fields.
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
from physics.simulator import simulate_positions_only
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS


# =============================================================================
# PyTorch Model
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """Transformer encoder for trajectories."""
    
    def __init__(self, hidden_dim=256, latent_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, 2)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_proj(x)
        return x


class ForceFieldDecoder(nn.Module):
    """Decode latent to wind field parameters."""
    
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Wind field outputs
        self.wind_center = nn.Linear(hidden_dim, 2)
        self.wind_size = nn.Linear(hidden_dim, 2)
        self.wind_direction = nn.Linear(hidden_dim, 2)
        self.wind_strength = nn.Linear(hidden_dim, 1)
        
    def forward(self, z):
        h = self.net(z)
        
        center = self.wind_center(h)
        size = torch.nn.functional.softplus(self.wind_size(h)) + 0.5  # Ensure positive
        direction = torch.tanh(self.wind_direction(h))
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        strength = torch.nn.functional.softplus(self.wind_strength(h)) * 10  # Scale to reasonable range
        
        return {
            'center': center,
            'size': size,
            'direction': direction,
            'strength': strength.squeeze(-1)
        }


class InverseModel(nn.Module):
    """Full model: trajectory -> force field."""
    
    def __init__(self, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(hidden_dim, latent_dim)
        self.decoder = ForceFieldDecoder(latent_dim, hidden_dim)
        
    def forward(self, trajectory):
        z = self.encoder(trajectory)
        params = self.decoder(z)
        return params


# =============================================================================
# Dataset
# =============================================================================

class TrajectoryDataset(Dataset):
    """Dataset of trajectories and their generating force fields."""
    
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'trajectory': torch.tensor(np.array(sample['trajectory']), dtype=torch.float32),
            'initial_position': torch.tensor(np.array(sample['initial_position']), dtype=torch.float32),
            'initial_velocity': torch.tensor(np.array(sample['initial_velocity']), dtype=torch.float32),
        }


# =============================================================================
# Physics Simulation (JAX on CPU for data, PyTorch for loss)
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
    
    return torch.stack(positions, dim=1)  # (batch, steps, 2)


def trajectory_loss(predicted_traj, target_traj):
    """MSE loss between trajectories."""
    return ((predicted_traj - target_traj) ** 2).mean()


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, dt=0.01, num_steps=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
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
        
        # Compute loss
        loss = trajectory_loss(pred_traj, trajectory)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device, dt=0.01, num_steps=100):
    """Evaluate model."""
    model.eval()
    total_loss = 0
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
            
            loss = trajectory_loss(pred_traj, trajectory)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# =============================================================================
# Data Generation
# =============================================================================

def generate_training_data(num_samples, seed=42):
    """Generate training data using JAX on CPU."""
    print(f"Generating {num_samples} training samples...")
    
    key = random.PRNGKey(seed)
    samples = []
    
    easy_level = DIFFICULTY_LEVELS['easy']
    
    for i in tqdm(range(num_samples), desc="Generating"):
        key, subkey = random.split(key)
        sample = generate_sample_at_difficulty(subkey, easy_level)
        samples.append(sample)
    
    return samples


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train TrajectoryForge with PyTorch')
    parser.add_argument('--samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--output', type=str, default='models/trained_model_pytorch.pt', help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("TRAJECTORYFORGE PYTORCH TRAINING")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Generate data
    samples = generate_training_data(args.samples, args.seed)
    
    # Split into train/val
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    
    print(f"\n📦 Data:")
    print(f"   Train: {len(train_samples)}")
    print(f"   Val: {len(val_samples)}")
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_samples)
    val_dataset = TrajectoryDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = InverseModel(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 Model: {n_params:,} parameters")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print("=" * 60)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
                'history': history,
            }, args.output)
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"best={best_val_loss:.4f}, time={elapsed/60:.1f}min")
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Model saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()

