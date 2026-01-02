#!/usr/bin/env python3 -u
"""
TrajectoryForge Multi-Field Training v2

Features:
- Checkpointing every N epochs (can resume!)
- Progress bars with tqdm
- Unbuffered output
- Multi-GPU support with PyTorch DistributedDataParallel (DDP)
- Signal handling for graceful shutdown
"""

import argparse
import os
import sys
import time
import signal
from pathlib import Path

# Unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm

# JAX for data generation only (CPU)
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random

from physics.state import PhysicsState, SimulationConfig
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS


# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    print("\n⚠️  Shutdown requested! Will save checkpoint after current epoch...")
    SHUTDOWN_REQUESTED = True


# =============================================================================
# Model Architecture
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """Transformer encoder for trajectories."""
    
    def __init__(self, hidden_dim=256, latent_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(2, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 300, hidden_dim) * 0.02)
        
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
    """Slot-based decoder for multiple force fields."""
    
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
        
        self.type_heads = nn.ModuleList([nn.Linear(slot_hidden, 5) for _ in range(num_slots)])
        self.center_heads = nn.ModuleList([nn.Linear(slot_hidden, 2) for _ in range(num_slots)])
        self.wind_heads = nn.ModuleList([nn.Linear(slot_hidden, 5) for _ in range(num_slots)])
        self.vortex_heads = nn.ModuleList([nn.Linear(slot_hidden, 2) for _ in range(num_slots)])
        self.point_heads = nn.ModuleList([nn.Linear(slot_hidden, 2) for _ in range(num_slots)])
    
    def forward(self, z):
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
    """Full model: trajectory -> multiple force fields."""
    
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
# Physics Simulation
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
# Dataset
# =============================================================================

class MultiFieldDataset(Dataset):
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

def train_epoch(model, dataloader, optimizer, device, num_steps=100, is_distributed=False):
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
        pred_traj = simulate_trajectory_multifield(init_pos, init_vel, slots, dt=0.01, num_steps=num_steps)
        
        traj_loss = F.mse_loss(pred_traj, trajectory)
        end_loss = F.mse_loss(pred_traj[:, -1], trajectory[:, -1])
        
        sparsity_loss = sum((1.0 - s['type_probs'][:, 0]).mean() for s in slots) / len(slots) * 0.1
        
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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            trajectory = batch['trajectory'].to(device)
            init_pos = batch['initial_position'].to(device)
            init_vel = batch['initial_velocity'].to(device)
            
            slots = model(trajectory)
            pred_traj = simulate_trajectory_multifield(init_pos, init_vel, slots, dt=0.01, num_steps=num_steps)
            
            loss = F.mse_loss(pred_traj, trajectory)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(num_samples_per_difficulty, seed=42, cache_path=None):
    """Generate or load cached training data."""
    
    # Check for cached data
    if cache_path and os.path.exists(cache_path):
        print(f"📂 Loading cached data from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        print(f"   Loaded {len(data)} samples")
        return data
    
    print(f"🔄 Generating {num_samples_per_difficulty * 4} samples...")
    
    key = random.PRNGKey(seed)
    all_samples = []
    
    for diff_name in ['easy', 'medium', 'hard', 'expert']:
        level = DIFFICULTY_LEVELS[diff_name]
        print(f"  {diff_name}...", flush=True)
        
        for i in tqdm(range(num_samples_per_difficulty), desc=f"  {diff_name}"):
            key, subkey = random.split(key)
            try:
                sample = generate_sample_at_difficulty(subkey, level)
                sample['difficulty'] = diff_name
                all_samples.append(sample)
            except:
                continue
    
    np.random.seed(seed)
    np.random.shuffle(all_samples)
    
    # Cache the data
    if cache_path:
        print(f"💾 Caching data to {cache_path}")
        torch.save(all_samples, cache_path)
    
    print(f"✅ Total: {len(all_samples)} samples")
    return all_samples


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, history, args):
    """Save training checkpoint."""
    # Handle DDP wrapped models
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'history': history,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path} (epoch {epoch})")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    """Load training checkpoint."""
    print(f"📂 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"   Resuming from epoch {checkpoint['epoch']}, best_val_loss={checkpoint['best_val_loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['best_val_loss'], checkpoint.get('history', {})


# =============================================================================
# Multi-GPU Setup
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-per-difficulty', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--num-slots', type=int, default=6)
    parser.add_argument('--output', type=str, default='models/multifield_model.pt')
    parser.add_argument('--checkpoint', type=str, default='models/multifield_checkpoint.pt')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--data-cache', type=str, default='models/training_data_cache.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    if is_main:
        print("=" * 60)
        print("TRAJECTORYFORGE MULTI-FIELD TRAINING v2")
        print("=" * 60)
        print(f"Distributed: {is_distributed}, World Size: {world_size}")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        if is_main:
            print(f"Device: {device} ({torch.cuda.get_device_name(local_rank)})")
    else:
        device = torch.device('cpu')
        if is_main:
            print(f"Device: CPU")
    
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Generate/load data (only on main process, then broadcast)
    if is_main:
        samples = generate_data(args.samples_per_difficulty, args.seed, args.data_cache)
    else:
        samples = None
    
    if is_distributed:
        # Broadcast data to all processes
        if rank == 0:
            data_list = [samples]
        else:
            data_list = [None]
        dist.broadcast_object_list(data_list, src=0)
        samples = data_list[0]
    
    # Split data
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    
    if is_main:
        print(f"\n📊 Data: {len(train_samples)} train, {len(val_samples)} val")
    
    # Create datasets and loaders
    train_dataset = MultiFieldDataset(train_samples, target_length=100)
    val_dataset = MultiFieldDataset(val_samples, target_length=100)
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = MultiFieldModel(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_slots=args.num_slots
    ).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"🔧 Model: {n_params:,} parameters, {args.num_slots} slots")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train': [], 'val': [], 'lr': []}
    
    if args.resume and os.path.exists(args.checkpoint):
        start_epoch, best_val_loss, history = load_checkpoint(
            args.checkpoint, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
    
    if is_main:
        print(f"\n🚀 Training epochs {start_epoch} to {args.epochs-1}")
        print("=" * 60)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if SHUTDOWN_REQUESTED:
            if is_main:
                print("\n🛑 Shutdown requested, saving checkpoint...")
                save_checkpoint(args.checkpoint, model, optimizer, scheduler, epoch, best_val_loss, history, args)
            break
        
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, is_distributed=is_distributed)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Record history
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss and is_main:
            best_val_loss = val_loss
            save_checkpoint(args.output, model, optimizer, scheduler, epoch, best_val_loss, history, args)
        
        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0 and is_main:
            save_checkpoint(args.checkpoint, model, optimizer, scheduler, epoch, best_val_loss, history, args)
        
        # Print progress
        if is_main:
            elapsed = time.time() - start_time
            eta = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1) if epoch > start_epoch else 0
            print(f"Epoch {epoch:3d}/{args.epochs}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"best={best_val_loss:.4f}, lr={current_lr:.2e}, "
                  f"time={elapsed/60:.1f}m, ETA={eta/60:.1f}m")
    
    # Final save
    if is_main:
        save_checkpoint(args.checkpoint, model, optimizer, scheduler, epoch, best_val_loss, history, args)
        
        elapsed = time.time() - start_time
        print("=" * 60)
        print("✅ TRAINING COMPLETE")
        print(f"   Time: {elapsed/60:.1f} minutes")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   Model: {args.output}")
        print(f"   Checkpoint: {args.checkpoint}")
        print("=" * 60)
    
    cleanup_distributed()


if __name__ == '__main__':
    main()

