#!/usr/bin/env python3
"""
TrajectoryForge Multi-Field Training

Trains a model that can predict MULTIPLE force fields of ALL types:
- Wind fields (rectangular, directional)
- Vortex fields (circular, rotational)
- Attractors (circular, pulls inward)
- Repellers (circular, pushes outward)

The model uses a slot-based architecture where it predicts N force field "slots",
each with a type classifier and type-specific parameters.
"""

import argparse
import os
import sys
import time
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
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS


# =============================================================================
# Model Architecture - Multi-Field Prediction
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
    """
    Slot-based decoder that predicts multiple force fields.
    
    Each slot outputs:
    - field_type: 5-class (none, wind, vortex, attractor, repeller)
    - center: (x, y) position
    - For wind: size (w, h), direction (dx, dy), strength
    - For vortex: radius, strength (signed for CW/CCW)
    - For point forces: radius, strength (positive=attractor, negative=repeller)
    """
    
    def __init__(self, latent_dim=128, hidden_dim=256, num_slots=6):
        super().__init__()
        self.num_slots = num_slots
        
        # Shared processing
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Per-slot heads
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
        
        # Output heads for each slot
        self.type_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 5)  # none, wind, vortex, attractor, repeller
            for _ in range(num_slots)
        ])
        
        self.center_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 2) for _ in range(num_slots)
        ])
        
        # Wind-specific: size (2), direction (2), strength (1) = 5
        self.wind_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 5) for _ in range(num_slots)
        ])
        
        # Vortex-specific: radius (1), strength (1) = 2
        self.vortex_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 2) for _ in range(num_slots)
        ])
        
        # Point force-specific: radius (1), strength (1) = 2
        self.point_heads = nn.ModuleList([
            nn.Linear(slot_hidden, 2) for _ in range(num_slots)
        ])
    
    def forward(self, z):
        """
        Returns dict with predictions for each slot.
        """
        batch_size = z.shape[0]
        h = self.shared(z)
        
        slots = []
        for i in range(self.num_slots):
            slot_h = self.slot_nets[i](h)
            
            # Field type logits
            type_logits = self.type_heads[i](slot_h)
            
            # Center position (scaled to reasonable range)
            center = self.center_heads[i](slot_h) * 3.0
            
            # Wind parameters
            wind_raw = self.wind_heads[i](slot_h)
            wind_size = F.softplus(wind_raw[:, :2]) + 0.5
            wind_dir = wind_raw[:, 2:4]
            wind_dir = wind_dir / (wind_dir.norm(dim=-1, keepdim=True) + 1e-8)
            wind_strength = F.softplus(wind_raw[:, 4:5]) * 5.0
            
            # Vortex parameters
            vortex_raw = self.vortex_heads[i](slot_h)
            vortex_radius = F.softplus(vortex_raw[:, 0:1]) + 0.5
            vortex_strength = vortex_raw[:, 1:2] * 5.0  # Can be negative
            
            # Point force parameters
            point_raw = self.point_heads[i](slot_h)
            point_radius = F.softplus(point_raw[:, 0:1]) + 0.5
            point_strength = point_raw[:, 1:2] * 5.0  # Positive=attractor, negative=repeller
            
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
# Physics Simulation (PyTorch - All Field Types)
# =============================================================================

def soft_indicator_box(pos, center, size, softness=0.1):
    """Soft rectangular indicator."""
    rel = (pos - center) / size
    scale = 1.0 / softness
    ind = torch.sigmoid(scale * (1.0 - torch.abs(rel)))
    return ind.prod(dim=-1, keepdim=True)


def soft_indicator_circle(pos, center, radius, softness=0.1):
    """Soft circular indicator."""
    dist = (pos - center).norm(dim=-1, keepdim=True)
    rel_dist = dist / radius
    scale = 1.0 / softness
    return torch.sigmoid(scale * (1.0 - rel_dist))


def compute_wind_force(pos, center, size, direction, strength):
    """Compute wind field force."""
    indicator = soft_indicator_box(pos, center, size)
    return indicator * direction * strength.unsqueeze(-1)


def compute_vortex_force(pos, center, radius, strength):
    """Compute vortex field force (perpendicular to radial)."""
    indicator = soft_indicator_circle(pos, center, radius)
    to_center = pos - center
    # Perpendicular direction (rotate 90 degrees)
    tangent = torch.stack([-to_center[..., 1], to_center[..., 0]], dim=-1)
    tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
    return indicator * tangent * strength.unsqueeze(-1)


def compute_point_force(pos, center, radius, strength):
    """Compute point force (attractor if strength > 0, repeller if < 0)."""
    indicator = soft_indicator_circle(pos, center, radius)
    to_center = center - pos
    dist = to_center.norm(dim=-1, keepdim=True)
    direction = to_center / (dist + 1e-8)
    falloff = 1.0 / (1.0 + dist * 0.5)
    return indicator * direction * strength.unsqueeze(-1) * falloff


def simulate_trajectory_multifield(init_pos, init_vel, slots, dt=0.01, num_steps=100):
    """
    Simulate trajectory with multiple force fields.
    
    slots: list of dicts with field parameters
    """
    batch_size = init_pos.shape[0]
    device = init_pos.device
    
    position = init_pos.clone()
    velocity = init_vel.clone()
    positions = [position.clone()]
    
    for _ in range(num_steps - 1):
        total_force = torch.zeros_like(position)
        
        for slot in slots:
            # Get field type probabilities
            type_probs = slot['type_probs']  # (batch, 5)
            
            # Compute force for each type, weighted by probability
            # Type 0 = none (no force)
            # Type 1 = wind
            wind_force = compute_wind_force(
                position, slot['center'], slot['wind_size'],
                slot['wind_direction'], slot['wind_strength']
            )
            
            # Type 2 = vortex
            vortex_force = compute_vortex_force(
                position, slot['center'], slot['vortex_radius'],
                slot['vortex_strength']
            )
            
            # Type 3 = attractor (positive strength)
            attractor_force = compute_point_force(
                position, slot['center'], slot['point_radius'],
                torch.abs(slot['point_strength'])
            )
            
            # Type 4 = repeller (negative strength)
            repeller_force = compute_point_force(
                position, slot['center'], slot['point_radius'],
                -torch.abs(slot['point_strength'])
            )
            
            # Weight by type probabilities
            # type_probs: (batch, 5) -> need (batch, 1) for each type
            slot_force = (
                type_probs[:, 1:2] * wind_force +
                type_probs[:, 2:3] * vortex_force +
                type_probs[:, 3:4] * attractor_force +
                type_probs[:, 4:5] * repeller_force
            )
            
            total_force = total_force + slot_force
        
        # Semi-implicit Euler
        velocity = velocity + total_force * dt
        velocity = velocity * 0.99  # Damping
        position = position + velocity * dt
        
        positions.append(position.clone())
    
    return torch.stack(positions, dim=1)


# =============================================================================
# Dataset
# =============================================================================

class MultiFieldDataset(Dataset):
    """Dataset for multi-field training."""
    
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
            'difficulty': sample.get('difficulty', 'unknown'),
        }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, num_steps=100):
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
        slots = model(trajectory)
        
        # Simulate with predicted fields
        pred_traj = simulate_trajectory_multifield(
            init_pos, init_vel, slots, dt=0.01, num_steps=num_steps
        )
        
        # Trajectory loss
        traj_loss = F.mse_loss(pred_traj, trajectory)
        
        # Endpoint loss (extra weight)
        end_loss = F.mse_loss(pred_traj[:, -1], trajectory[:, -1])
        
        # Sparsity loss - encourage using fewer fields
        sparsity_loss = 0
        for slot in slots:
            # Encourage "none" type (index 0)
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
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, num_steps=100):
    """Evaluate model."""
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


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(num_samples_per_difficulty, seed=42):
    """Generate training data from all difficulties."""
    print(f"Generating data...")
    
    key = random.PRNGKey(seed)
    all_samples = []
    
    for diff_name in ['easy', 'medium', 'hard', 'expert']:
        level = DIFFICULTY_LEVELS[diff_name]
        print(f"  {diff_name}...")
        
        for i in tqdm(range(num_samples_per_difficulty), desc=f"  {diff_name}", leave=False):
            key, subkey = random.split(key)
            try:
                sample = generate_sample_at_difficulty(subkey, level)
                sample['difficulty'] = diff_name
                all_samples.append(sample)
            except:
                continue
        
        print(f"    {diff_name}: {len([s for s in all_samples if s.get('difficulty') == diff_name])} samples")
    
    np.random.seed(seed)
    np.random.shuffle(all_samples)
    print(f"Total: {len(all_samples)} samples")
    return all_samples


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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("TRAJECTORYFORGE MULTI-FIELD TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Generate data
    samples = generate_data(args.samples_per_difficulty, args.seed)
    
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_dataset = MultiFieldDataset(train_samples, target_length=100)
    val_dataset = MultiFieldDataset(val_samples, target_length=100)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    
    # Create model
    model = MultiFieldModel(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_slots=args.num_slots
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Slots: {args.num_slots} (can predict up to {args.num_slots} force fields)")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'val_loss': val_loss,
                'history': history,
            }, args.output)
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"best={best_val_loss:.4f}, time={elapsed/60:.1f}min")
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()

