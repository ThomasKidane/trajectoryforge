#!/usr/bin/env python3
"""
Evaluate the multi-field model checkpoint and generate visualizations.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
import os

# JAX for data generation (CPU only)
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random

from physics.state import PhysicsState, SimulationConfig
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS


# =============================================================================
# Model Architecture (must match training)
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
# Physics Simulation (PyTorch)
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
    force = direction * strength.unsqueeze(-1) * indicator
    return force


def compute_vortex_force(pos, center, radius, strength):
    diff = pos - center
    dist = diff.norm(dim=-1, keepdim=True) + 1e-8
    tangent = torch.stack([-diff[..., 1], diff[..., 0]], dim=-1)
    tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
    indicator = soft_indicator_circle(pos, center, radius)
    strength_expanded = strength.unsqueeze(-1) if strength.dim() == 1 else strength
    force = tangent * strength_expanded * indicator
    return force


def compute_point_force(pos, center, radius, strength):
    diff = center - pos
    dist = diff.norm(dim=-1, keepdim=True) + 1e-8
    direction = diff / dist
    indicator = soft_indicator_circle(pos, center, radius)
    strength_expanded = strength.unsqueeze(-1) if strength.dim() == 1 else strength
    force = direction * strength_expanded * indicator
    return force


def simulate_trajectory_multifield(init_pos, init_vel, slots, num_steps=100, dt=0.02):
    """Simulate trajectory with multiple force fields."""
    batch_size = init_pos.shape[0]
    device = init_pos.device
    
    positions = [init_pos]
    pos = init_pos.clone()
    vel = init_vel.clone()
    
    for step in range(num_steps - 1):
        total_force = torch.zeros_like(pos)
        
        for slot in slots:
            type_probs = slot['type_probs']
            center = slot['center']
            
            # Wind force (type index 1)
            wind_prob = type_probs[:, 1:2]
            wind_force = compute_wind_force(
                pos, center, slot['wind_size'],
                slot['wind_direction'], slot['wind_strength']
            )
            total_force = total_force + wind_prob * wind_force
            
            # Vortex force (type index 2)
            vortex_prob = type_probs[:, 2:3]
            vortex_force = compute_vortex_force(
                pos, center, slot['vortex_radius'], slot['vortex_strength']
            )
            total_force = total_force + vortex_prob * vortex_force
            
            # Attractor force (type index 3)
            attractor_prob = type_probs[:, 3:4]
            attractor_force = compute_point_force(
                pos, center, slot['point_radius'], slot['point_strength']
            )
            total_force = total_force + attractor_prob * attractor_force
            
            # Repeller force (type index 4) - negative strength
            repeller_prob = type_probs[:, 4:5]
            repeller_force = compute_point_force(
                pos, center, slot['point_radius'], -slot['point_strength']
            )
            total_force = total_force + repeller_prob * repeller_force
        
        vel = vel + total_force * dt
        pos = pos + vel * dt
        positions.append(pos)
    
    return torch.stack(positions, dim=1)


# =============================================================================
# Visualization
# =============================================================================

FIELD_COLORS = {
    'none': '#888888',
    'wind': '#3498db',
    'vortex': '#9b59b6',
    'attractor': '#27ae60',
    'repeller': '#e74c3c'
}

FIELD_NAMES = ['none', 'wind', 'vortex', 'attractor', 'repeller']


def draw_force_field(ax, slot, alpha=0.5):
    """Draw a force field on the given axes."""
    type_idx = slot['type_probs'].argmax().item()
    type_name = FIELD_NAMES[type_idx]
    
    if type_name == 'none':
        return
    
    center = slot['center'].detach().cpu().numpy()
    color = FIELD_COLORS[type_name]
    
    if type_name == 'wind':
        size = slot['wind_size'].detach().cpu().numpy()
        direction = slot['wind_direction'].detach().cpu().numpy()
        strength = slot['wind_strength'].item()
        
        rect = Rectangle(
            (center[0] - size[0], center[1] - size[1]),
            size[0] * 2, size[1] * 2,
            fill=True, facecolor=color, edgecolor='white',
            alpha=alpha * 0.3, linewidth=2
        )
        ax.add_patch(rect)
        
        # Draw arrow for direction
        ax.arrow(center[0], center[1], 
                direction[0] * 0.5, direction[1] * 0.5,
                head_width=0.15, head_length=0.1,
                fc=color, ec='white', linewidth=1.5, alpha=alpha)
        
    elif type_name == 'vortex':
        radius = slot['vortex_radius'].item()
        strength = slot['vortex_strength'].item()
        
        circle = Circle(center, radius, fill=True, facecolor=color,
                       edgecolor='white', alpha=alpha * 0.3, linewidth=2)
        ax.add_patch(circle)
        
        # Draw rotation indicator
        rotation = '↻' if strength > 0 else '↺'
        ax.text(center[0], center[1], rotation, fontsize=16, ha='center', va='center',
               color='white', fontweight='bold')
        
    elif type_name in ['attractor', 'repeller']:
        radius = slot['point_radius'].item()
        
        circle = Circle(center, radius, fill=True, facecolor=color,
                       edgecolor='white', alpha=alpha * 0.3, linewidth=2)
        ax.add_patch(circle)
        
        # Draw symbol
        symbol = '⊕' if type_name == 'attractor' else '⊖'
        ax.text(center[0], center[1], symbol, fontsize=14, ha='center', va='center',
               color='white', fontweight='bold')


def generate_test_samples(num_samples=20, seed=42):
    """Generate test samples across all difficulties."""
    key = random.PRNGKey(seed)
    samples = []
    
    difficulties = ['trivial', 'easy', 'medium', 'hard']
    samples_per_diff = num_samples // len(difficulties)
    
    for diff_name in difficulties:
        diff = DIFFICULTY_LEVELS[diff_name]  # Get the DifficultyLevel object
        for i in range(samples_per_diff):
            key, subkey = random.split(key)
            sample = generate_sample_at_difficulty(subkey, diff)
            sample['difficulty'] = diff_name
            samples.append(sample)
    
    return samples


def evaluate_model(model, samples, device='cpu'):
    """Evaluate model on samples and return predictions."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for sample in samples:
            # Get target trajectory
            target_traj = np.array(sample['trajectory'])
            target_tensor = torch.tensor(target_traj, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get initial conditions
            init_pos = torch.tensor(target_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
            init_vel = torch.tensor(np.array(sample['initial_velocity']), dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict force fields
            slots = model(target_tensor)
            
            # Simulate with predicted fields
            pred_traj = simulate_trajectory_multifield(
                init_pos, init_vel, slots, 
                num_steps=len(target_traj), dt=0.02
            )
            
            # Calculate error
            pred_np = pred_traj[0].cpu().numpy()
            error = np.mean(np.sqrt(np.sum((pred_np - target_traj[:len(pred_np)])**2, axis=1)))
            
            results.append({
                'target': target_traj,
                'predicted': pred_np,
                'slots': [{k: v[0] if isinstance(v, torch.Tensor) else v for k, v in slot.items()} for slot in slots],
                'error': error,
                'difficulty': sample.get('difficulty', 'unknown')
            })
    
    return results


def create_evaluation_grid(results, output_path, epoch, num_cols=5):
    """Create a grid visualization of predictions."""
    num_samples = len(results)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    fig.suptitle(f'Multi-Field Model Predictions (Epoch {epoch})', fontsize=16, fontweight='bold')
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        row, col = idx // num_cols, idx % num_cols
        ax = axes[row, col]
        
        target = result['target']
        pred = result['predicted']
        
        # Draw force fields
        for slot in result['slots']:
            draw_force_field(ax, slot, alpha=0.6)
        
        # Draw trajectories
        ax.plot(target[:, 0], target[:, 1], 'g-', linewidth=2, label='Target', alpha=0.8)
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        
        # Mark start and end
        ax.scatter(target[0, 0], target[0, 1], c='green', s=100, marker='o', zorder=5, edgecolors='white')
        ax.scatter(target[-1, 0], target[-1, 1], c='green', s=100, marker='*', zorder=5, edgecolors='white')
        
        ax.set_xlim(-3, 5)
        ax.set_ylim(-3, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{result['difficulty'].capitalize()}\nError: {result['error']:.3f}", fontsize=10)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide empty subplots
    for idx in range(num_samples, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation grid to {output_path}")


def create_error_distribution(results, output_path):
    """Create error distribution plot by difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall error distribution
    errors = [r['error'] for r in results]
    axes[0].hist(errors, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    axes[0].axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}')
    axes[0].set_xlabel('Trajectory Error', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Overall Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error by difficulty
    difficulties = ['trivial', 'easy', 'medium', 'hard']
    diff_errors = {d: [r['error'] for r in results if r['difficulty'] == d] for d in difficulties}
    
    positions = range(len(difficulties))
    box_data = [diff_errors[d] for d in difficulties]
    
    bp = axes[1].boxplot(box_data, positions=positions, patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([d.capitalize() for d in difficulties])
    axes[1].set_xlabel('Difficulty Level', fontsize=12)
    axes[1].set_ylabel('Trajectory Error', fontsize=12)
    axes[1].set_title('Error by Difficulty Level', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    for i, d in enumerate(difficulties):
        if diff_errors[d]:
            mean_val = np.mean(diff_errors[d])
            axes[1].scatter(i, mean_val, color='black', s=50, zorder=5, marker='D')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error distribution to {output_path}")


def create_best_worst_comparison(results, output_path, num_each=4):
    """Show best and worst predictions."""
    sorted_results = sorted(results, key=lambda x: x['error'])
    best = sorted_results[:num_each]
    worst = sorted_results[-num_each:]
    
    fig, axes = plt.subplots(2, num_each, figsize=(4 * num_each, 8))
    fig.suptitle('Best vs Worst Predictions', fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(best):
        ax = axes[0, idx]
        target = result['target']
        pred = result['predicted']
        
        for slot in result['slots']:
            draw_force_field(ax, slot, alpha=0.5)
        
        ax.plot(target[:, 0], target[:, 1], 'g-', linewidth=2, alpha=0.8)
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, alpha=0.8)
        ax.scatter(target[0, 0], target[0, 1], c='green', s=80, marker='o', zorder=5)
        
        ax.set_xlim(-3, 5)
        ax.set_ylim(-3, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"BEST #{idx+1}\n{result['difficulty'].capitalize()}, Error: {result['error']:.3f}", 
                    fontsize=10, color='green')
    
    for idx, result in enumerate(worst):
        ax = axes[1, idx]
        target = result['target']
        pred = result['predicted']
        
        for slot in result['slots']:
            draw_force_field(ax, slot, alpha=0.5)
        
        ax.plot(target[:, 0], target[:, 1], 'g-', linewidth=2, alpha=0.8)
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, alpha=0.8)
        ax.scatter(target[0, 0], target[0, 1], c='green', s=80, marker='o', zorder=5)
        
        ax.set_xlim(-3, 5)
        ax.set_ylim(-3, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"WORST #{idx+1}\n{result['difficulty'].capitalize()}, Error: {result['error']:.3f}", 
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved best/worst comparison to {output_path}")


def main():
    # Paths - use final trained model
    checkpoint_path = Path(__file__).parent.parent / 'models' / 'multifield_model_final.pt'
    output_dir = Path(__file__).parent.parent / 'models'
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model config
    num_slots = checkpoint.get('num_slots', 6)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    latent_dim = checkpoint.get('latent_dim', 128)
    epoch = checkpoint.get('epoch', 'unknown')
    
    print(f"Model config: hidden_dim={hidden_dim}, latent_dim={latent_dim}, num_slots={num_slots}")
    print(f"Checkpoint from epoch {epoch}")
    
    # Create model
    model = MultiFieldModel(hidden_dim=hidden_dim, latent_dim=latent_dim, num_slots=num_slots)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    
    # Generate test samples
    print("Generating test samples...")
    samples = generate_test_samples(num_samples=20, seed=12345)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, samples)
    
    # Calculate stats
    errors = [r['error'] for r in results]
    print(f"\n📊 Evaluation Results:")
    print(f"   Mean Error: {np.mean(errors):.4f}")
    print(f"   Median Error: {np.median(errors):.4f}")
    print(f"   Min Error: {np.min(errors):.4f}")
    print(f"   Max Error: {np.max(errors):.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_evaluation_grid(results, output_dir / 'multifield_eval_grid.png', epoch)
    create_error_distribution(results, output_dir / 'multifield_eval_errors.png')
    create_best_worst_comparison(results, output_dir / 'multifield_eval_best_worst.png')
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
