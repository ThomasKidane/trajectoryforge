#!/usr/bin/env python3
"""
TrajectoryForge Multi-Field Training V3 - Fully Random

Key changes from v2:
- No difficulty levels - everything is randomly generated
- Larger batch sizes to utilize 48GB VRAM
- Fixed number of force fields per sample (configurable)
- Pure random initialization for all field parameters
"""

import argparse
import os
import sys
import time
import signal
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# JAX for data generation only
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField, VortexField, PointForce
from physics.simulator import simulate_trajectory


# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n⚠️  Shutdown requested, will save checkpoint after current epoch...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# Model Architecture (same as v2)
# =============================================================================

class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(2, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x


class MultiFieldDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_slots=6):
        super().__init__()
        self.num_slots = num_slots
        
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        self.slot_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
            )
            for _ in range(num_slots)
        ])
        
        slot_hidden = hidden_dim // 2
        
        self.type_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 5) for _ in range(num_slots)
        ])
        self.center_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 2) for _ in range(num_slots)
        ])
        self.wind_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 5) for _ in range(num_slots)
        ])
        self.vortex_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 2) for _ in range(num_slots)
        ])
        self.point_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 2) for _ in range(num_slots)
        ])
    
    def forward(self, z):
        batch_size = z.shape[0]
        h = self.shared(z)
        
        slots = []
        for i in range(self.num_slots):
            slot_h = self.slot_nets[i](h)
            
            type_logits = self.type_heads[i](slot_h)
            center = self.center_heads[i](slot_h) * 3.0
            
            wind_raw = self.wind_heads[i](slot_h)
            wind_size = F.softplus(wind_raw[:, :2]) + 0.5
            wind_dir = wind_raw[:, 2:4]
            wind_dir = wind_dir / (wind_dir.norm(dim=-1, keepdim=True) + 1e-8)
            wind_strength = F.softplus(wind_raw[:, 4:5]) * 5.0
            
            vortex_raw = self.vortex_heads[i](slot_h)
            vortex_radius = F.softplus(vortex_raw[:, 0:1]) + 0.5
            vortex_strength = vortex_raw[:, 1:2] * 5.0
            
            point_raw = self.point_heads[i](slot_h)
            point_radius = F.softplus(point_raw[:, 0:1]) + 0.5
            point_strength = point_raw[:, 1:2] * 5.0
            
            slots.append({
                'type_logits': type_logits,
                'type_probs': F.softmax(type_logits, dim=-1),
                'center': center,
                'wind_size': wind_size,
                'wind_direction': wind_dir,
                'wind_strength': wind_strength.squeeze(-1),
                'vortex_radius': vortex_radius.squeeze(-1),
                'vortex_strength': vortex_strength.squeeze(-1),
                'point_radius': point_radius.squeeze(-1),
                'point_strength': point_strength.squeeze(-1),
            })
        
        return slots


class MultiFieldModel(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128, num_slots=6):
        super().__init__()
        self.encoder = TrajectoryEncoder(hidden_dim, latent_dim)
        self.decoder = MultiFieldDecoder(latent_dim, hidden_dim, num_slots)
        self.num_slots = num_slots
        
    def forward(self, trajectory):
        z = self.encoder(trajectory)
        slots = self.decoder(z)
        return slots


# =============================================================================
# Physics Simulation (same as v2)
# =============================================================================

def soft_indicator_box(pos, center, size, softness=0.1):
    rel = (pos - center) / (size + 1e-8)
    scale = 1.0 / softness
    ind = torch.sigmoid(scale * (1.0 - torch.abs(rel)))
    return ind.prod(dim=-1, keepdim=True)

def soft_indicator_circle(pos, center, radius, softness=0.1):
    dist = (pos - center).norm(dim=-1, keepdim=True)
    radius_expanded = radius.unsqueeze(-1) if radius.dim() == 1 else radius
    rel_dist = dist / (radius_expanded + 1e-8)
    scale = 1.0 / softness
    return torch.sigmoid(scale * (1.0 - rel_dist))

def compute_wind_force(pos, center, size, direction, strength):
    indicator = soft_indicator_box(pos, center, size)
    strength_expanded = strength.unsqueeze(-1) if strength.dim() == 1 else strength
    return indicator * direction * strength_expanded

def compute_vortex_force(pos, center, radius, strength):
    indicator = soft_indicator_circle(pos, center, radius)
    to_center = pos - center
    tangent = torch.stack([-to_center[..., 1], to_center[..., 0]], dim=-1)
    tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
    strength_expanded = strength.unsqueeze(-1) if strength.dim() == 1 else strength
    return indicator * tangent * strength_expanded

def compute_point_force(pos, center, radius, strength):
    indicator = soft_indicator_circle(pos, center, radius)
    to_center = center - pos
    dist = to_center.norm(dim=-1, keepdim=True)
    direction = to_center / (dist + 1e-8)
    falloff = 1.0 / (1.0 + dist * 0.5)
    strength_expanded = strength.unsqueeze(-1) if strength.dim() == 1 else strength
    return indicator * direction * strength_expanded * falloff

def simulate_trajectory_multifield(init_pos, init_vel, slots, dt=0.01, num_steps=100):
    batch_size = init_pos.shape[0]
    device = init_pos.device
    
    position = init_pos.clone()
    velocity = init_vel.clone()
    positions = [position.clone()]
    
    for _ in range(num_steps - 1):
        total_force = torch.zeros_like(position)
        
        for slot in slots:
            type_probs = slot['type_probs']
            
            wind_force = compute_wind_force(
                position, slot['center'], slot['wind_size'],
                slot['wind_direction'], slot['wind_strength']
            )
            vortex_force = compute_vortex_force(
                position, slot['center'], slot['vortex_radius'],
                slot['vortex_strength']
            )
            attractor_force = compute_point_force(
                position, slot['center'], slot['point_radius'],
                torch.abs(slot['point_strength'])
            )
            repeller_force = compute_point_force(
                position, slot['center'], slot['point_radius'],
                -torch.abs(slot['point_strength'])
            )
            
            slot_force = (
                type_probs[:, 1:2] * wind_force +
                type_probs[:, 2:3] * vortex_force +
                type_probs[:, 3:4] * attractor_force +
                type_probs[:, 4:5] * repeller_force
            )
            
            total_force = total_force + slot_force
        
        velocity = velocity + total_force * dt
        velocity = velocity * 0.99
        position = position + velocity * dt
        
        positions.append(position.clone())
    
    return torch.stack(positions, dim=1)


# =============================================================================
# FULLY RANDOM Data Generation (V3 - No Difficulty Levels)
# =============================================================================

def generate_random_sample(key, num_fields_range=(1, 4), num_steps=100, dt=0.02):
    """
    Generate a sample with FULLY RANDOM force fields.
    No difficulty levels - pure randomness for all parameters.
    """
    keys = random.split(key, 10)
    
    # Random initial position in [-3, 3] x [-3, 3]
    init_pos = random.uniform(keys[0], (2,), minval=-2.5, maxval=2.5)
    
    # Random initial velocity (can be zero or moving)
    vel_scale = random.uniform(keys[1], (), minval=0.0, maxval=2.0)
    vel_dir = random.uniform(keys[2], (2,), minval=-1.0, maxval=1.0)
    vel_dir = vel_dir / (jnp.linalg.norm(vel_dir) + 1e-8)
    init_vel = vel_dir * vel_scale
    
    # Random number of fields
    num_fields = int(random.randint(keys[3], (), num_fields_range[0], num_fields_range[1] + 1))
    
    # Generate random fields
    fields = []
    for i in range(num_fields):
        field_keys = random.split(keys[4 + i], 8)
        
        # Random field type (equal probability)
        field_type = int(random.randint(field_keys[0], (), 0, 4))
        
        # Random center position
        center = random.uniform(field_keys[1], (2,), minval=-3.0, maxval=3.0)
        
        if field_type == 0:  # Wind
            size = random.uniform(field_keys[2], (2,), minval=0.5, maxval=3.0)
            direction = random.uniform(field_keys[3], (2,), minval=-1.0, maxval=1.0)
            direction = direction / (jnp.linalg.norm(direction) + 1e-8)
            strength = random.uniform(field_keys[4], (), minval=1.0, maxval=8.0)
            fields.append(WindField(
                center=tuple(center.tolist()),
                size=tuple(size.tolist()),
                direction=tuple(direction.tolist()),
                strength=float(strength)
            ))
        
        elif field_type == 1:  # Vortex
            radius = float(random.uniform(field_keys[2], (), minval=0.5, maxval=2.5))
            strength = random.uniform(field_keys[3], (), minval=-6.0, maxval=6.0)
            fields.append(VortexField(
                center=tuple(center.tolist()),
                radius=radius,
                strength=float(strength)
            ))
        
        elif field_type == 2:  # Attractor
            radius = float(random.uniform(field_keys[2], (), minval=0.5, maxval=3.0))
            strength = float(random.uniform(field_keys[3], (), minval=2.0, maxval=10.0))
            fields.append(PointForce(
                center=tuple(center.tolist()),
                radius=radius,
                strength=strength,
                falloff_power=2.0
            ))
        
        else:  # Repeller
            radius = float(random.uniform(field_keys[2], (), minval=0.5, maxval=3.0))
            strength = float(random.uniform(field_keys[3], (), minval=-10.0, maxval=-2.0))
            fields.append(PointForce(
                center=tuple(center.tolist()),
                radius=radius,
                strength=strength,
                falloff_power=2.0
            ))
    
    # Simulate trajectory
    config = SimulationConfig(dt=dt, num_steps=num_steps, bounds=(-5.0, 5.0))
    init_state = PhysicsState(
        position=jnp.array(init_pos),
        velocity=jnp.array(init_vel),
        time=0.0
    )
    
    traj_data = simulate_trajectory(init_state, fields, config)
    trajectory = traj_data.positions
    
    return {
        'trajectory': np.array(trajectory),
        'initial_position': np.array(init_pos),
        'initial_velocity': np.array(init_vel),
        'num_fields': num_fields,
        'field_types': [type(f).__name__ for f in fields],
    }


def generate_data_random(num_samples, num_fields_range=(1, 4), seed=42, num_steps=100):
    """Generate fully random training data."""
    print(f"Generating {num_samples} random samples...")
    print(f"  Fields per sample: {num_fields_range[0]}-{num_fields_range[1]}")
    
    key = random.PRNGKey(seed)
    samples = []
    
    for i in tqdm(range(num_samples), desc="Generating"):
        key, subkey = random.split(key)
        try:
            sample = generate_random_sample(subkey, num_fields_range, num_steps=num_steps)
            samples.append(sample)
        except Exception as e:
            continue
    
    np.random.seed(seed)
    np.random.shuffle(samples)
    
    # Stats
    field_counts = {}
    type_counts = {}
    for s in samples:
        n = s['num_fields']
        field_counts[n] = field_counts.get(n, 0) + 1
        for t in s['field_types']:
            type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\nGenerated {len(samples)} samples")
    print(f"Field count distribution: {dict(sorted(field_counts.items()))}")
    print(f"Field type distribution: {dict(sorted(type_counts.items()))}")
    
    return samples


# =============================================================================
# Dataset
# =============================================================================

class RandomFieldDataset(Dataset):
    def __init__(self, samples, target_length=100):
        self.samples = samples
        self.target_length = target_length
        
    def __len__(self):
        return len(self.samples)
    
    def _resample(self, traj, target_len):
        current_len = len(traj)
        if current_len == target_len:
            return traj
        old_idx = np.linspace(0, current_len - 1, current_len)
        new_idx = np.linspace(0, current_len - 1, target_len)
        new_traj = np.zeros((target_len, 2))
        new_traj[:, 0] = np.interp(new_idx, old_idx, traj[:, 0])
        new_traj[:, 1] = np.interp(new_idx, old_idx, traj[:, 1])
        return new_traj
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        traj = np.array(sample['trajectory'])
        traj = self._resample(traj, self.target_length)
        
        return {
            'trajectory': torch.tensor(traj, dtype=torch.float32),
            'initial_position': torch.tensor(np.array(sample['initial_position']), dtype=torch.float32),
            'initial_velocity': torch.tensor(np.array(sample['initial_velocity']), dtype=torch.float32),
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, num_steps=100):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        trajectory = batch['trajectory'].to(device)
        init_pos = batch['initial_position'].to(device)
        init_vel = batch['initial_velocity'].to(device)
        
        optimizer.zero_grad()
        
        slots = model(trajectory)
        pred_traj = simulate_trajectory_multifield(
            init_pos, init_vel, slots, dt=0.01, num_steps=num_steps
        )
        
        traj_loss = F.mse_loss(pred_traj, trajectory)
        end_loss = F.mse_loss(pred_traj[:, -1], trajectory[:, -1])
        
        sparsity_loss = 0
        for slot in slots:
            sparsity_loss += (1.0 - slot['type_probs'][:, 0]).mean()
        sparsity_loss = sparsity_loss / len(slots) * 0.1
        
        loss = traj_loss + 0.5 * end_loss + sparsity_loss
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, num_steps=100):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            trajectory = batch['trajectory'].to(device)
            init_pos = batch['initial_position'].to(device)
            init_vel = batch['initial_velocity'].to(device)
            
            slots = model(trajectory)
            pred_traj = simulate_trajectory_multifield(
                init_pos, init_vel, slots, dt=0.01, num_steps=num_steps
            )
            
            loss = F.mse_loss(pred_traj, trajectory)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, args, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args),
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TrajectoryForge V3 - Fully Random Training')
    parser.add_argument('--num-samples', type=int, default=50000, help='Total training samples')
    parser.add_argument('--min-fields', type=int, default=1, help='Min fields per sample')
    parser.add_argument('--max-fields', type=int, default=8, help='Max fields per sample')
    parser.add_argument('--num-steps', type=int, default=300, help='Number of timesteps per trajectory')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (increase for more VRAM usage)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num-slots', type=int, default=10, help='Number of prediction slots')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRAJECTORYFORGE V3 - FULLY RANDOM TRAINING")
    print("=" * 70)
    print(f"Samples: {args.num_samples}")
    print(f"Fields per sample: {args.min_fields}-{args.max_fields}")
    print(f"Timesteps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Slots: {args.num_slots}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Generate data
    cache_path = Path(args.output_dir) / f'random_data_cache_{args.num_samples}_steps{args.num_steps}.pt'
    if cache_path.exists():
        print(f"\nLoading cached data from {cache_path}...")
        samples = torch.load(cache_path, weights_only=False)
    else:
        samples = generate_data_random(
            args.num_samples, 
            num_fields_range=(args.min_fields, args.max_fields),
            seed=args.seed,
            num_steps=args.num_steps
        )
        print(f"Caching data to {cache_path}...")
        torch.save(samples, cache_path)
    
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_dataset = RandomFieldDataset(train_samples, target_length=args.num_steps)
    val_dataset = RandomFieldDataset(val_samples, target_length=args.num_steps)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        num_workers=0, pin_memory=True
    )
    
    # Create model
    model = MultiFieldModel(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_slots=args.num_slots
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if shutdown_requested:
            print(f"\n⚠️  Saving final checkpoint before shutdown...")
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss, args,
                Path(args.output_dir) / 'v3_interrupted.pt'
            )
            break
        
        train_loss = train_epoch(model, train_loader, optimizer, device, num_steps=args.num_steps)
        val_loss = evaluate(model, val_loader, device, num_steps=args.num_steps)
        scheduler.step()
        
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]
        
        # Progress output
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed/60:.1f}min")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss, args,
                Path(args.output_dir) / 'v3_best.pt'
            )
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss, args,
                Path(args.output_dir) / f'v3_checkpoint_ep{epoch}.pt'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, train_loss, val_loss, args,
        Path(args.output_dir) / 'v3_final.pt'
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Models saved in: {args.output_dir}/")


if __name__ == '__main__':
    main()

