#!/usr/bin/env python3
"""
Evaluate the trained PyTorch model and visualize predictions.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# For data generation we use JAX on CPU
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp
from jax import random

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField
from physics.simulator import simulate_positions_only
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS


# =============================================================================
# PyTorch Model (same as training)
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """Larger Transformer encoder for trajectories."""
    
    def __init__(self, hidden_dim=512, latent_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True,
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
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
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
        strength = torch.nn.functional.softplus(self.wind_strength(h)) * 10
        return {'center': center, 'size': size, 'direction': direction, 'strength': strength.squeeze(-1)}


class InverseModel(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(hidden_dim, latent_dim)
        self.decoder = ForceFieldDecoder(latent_dim, hidden_dim)
        
    def forward(self, trajectory):
        z = self.encoder(trajectory)
        params = self.decoder(z)
        return params


# =============================================================================
# Simulation
# =============================================================================

def resample_trajectory(traj, target_len=100):
    """Resample trajectory to target length using linear interpolation."""
    current_len = len(traj)
    if current_len == target_len:
        return traj
    
    old_indices = np.linspace(0, current_len - 1, current_len)
    new_indices = np.linspace(0, current_len - 1, target_len)
    
    new_traj = np.zeros((target_len, 2))
    new_traj[:, 0] = np.interp(new_indices, old_indices, traj[:, 0])
    new_traj[:, 1] = np.interp(new_indices, old_indices, traj[:, 1])
    
    return new_traj


def simulate_with_jax(init_pos, init_vel, wind_center, wind_size, wind_direction, wind_strength):
    """Simulate using JAX physics engine."""
    config = SimulationConfig(dt=0.01, num_steps=100)
    
    wind = WindField(
        center=jnp.array(wind_center),
        size=jnp.array(wind_size),
        direction=jnp.array(wind_direction),
        strength=float(wind_strength),
        softness=0.1,
    )
    
    state = PhysicsState(
        position=jnp.array(init_pos),
        velocity=jnp.array(init_vel),
        time=0.0,
    )
    
    trajectory = simulate_positions_only(state, [wind], config)
    return np.array(trajectory)


# =============================================================================
# Main
# =============================================================================

def main():
    # Load model - use the full trained model
    model_path = project_root / 'models' / 'trained_model_full.pt'
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    
    model = InverseModel(hidden_dim=args['hidden_dim'], latent_dim=args['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded! Best val loss: {checkpoint['val_loss']:.6f}")
    print(f"Parameters: hidden_dim={args['hidden_dim']}, latent_dim={args['latent_dim']}")
    
    # Generate test samples from ALL difficulty levels
    print("\nGenerating 20 test samples (5 per difficulty)...")
    key = random.PRNGKey(999)  # Different seed from training
    test_samples = []
    
    for diff_name in ['easy', 'medium', 'hard', 'expert']:
        level = DIFFICULTY_LEVELS[diff_name]
        for i in range(5):
            key, subkey = random.split(key)
            try:
                sample = generate_sample_at_difficulty(subkey, level)
                sample['difficulty'] = diff_name
                test_samples.append(sample)
            except:
                continue
    
    # Evaluate and collect results
    print("Evaluating model predictions...")
    results = []
    
    with torch.no_grad():
        for sample in test_samples:
            # Get original trajectory and resample to 100 steps
            orig_traj = np.array(sample['trajectory'])
            orig_traj = resample_trajectory(orig_traj, 100)
            init_pos = np.array(sample['initial_position'])
            init_vel = np.array(sample['initial_velocity'])
            
            # Model prediction
            traj_tensor = torch.tensor(orig_traj, dtype=torch.float32).unsqueeze(0)
            params = model(traj_tensor)
            
            pred_center = params['center'][0].numpy()
            pred_size = params['size'][0].numpy()
            pred_direction = params['direction'][0].numpy()
            pred_strength = params['strength'][0].item()
            
            # Simulate with predicted parameters
            pred_traj = simulate_with_jax(
                init_pos, init_vel,
                pred_center, pred_size, pred_direction, pred_strength
            )
            
            # Compute error
            error = np.mean((orig_traj - pred_traj) ** 2)
            
            results.append({
                'original': orig_traj,
                'predicted': pred_traj,
                'init_pos': init_pos,
                'init_vel': init_vel,
                'pred_params': {
                    'center': pred_center,
                    'size': pred_size,
                    'direction': pred_direction,
                    'strength': pred_strength,
                },
                'error': error,
            })
    
    errors = [r['error'] for r in results]
    print(f"\nError Statistics:")
    print(f"  Mean: {np.mean(errors):.6f}")
    print(f"  Std:  {np.std(errors):.6f}")
    print(f"  Min:  {np.min(errors):.6f}")
    print(f"  Max:  {np.max(errors):.6f}")
    
    # ==========================================================================
    # Visualization 1: 20-sample grid
    # ==========================================================================
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, (ax, result) in enumerate(zip(axes, results)):
        orig = result['original']
        pred = result['predicted']
        error = result['error']
        
        ax.plot(orig[:, 0], orig[:, 1], 'b-', linewidth=2, alpha=0.8, label='Original')
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, alpha=0.8, label='Predicted')
        ax.plot(orig[0, 0], orig[0, 1], 'go', markersize=10, zorder=5)
        ax.plot(orig[-1, 0], orig[-1, 1], 'b^', markersize=8, zorder=5)
        ax.plot(pred[-1, 0], pred[-1, 1], 'r^', markersize=8, zorder=5)
        
        ax.set_title(f'Sample {i+1}\nError: {error:.4f}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Model Evaluation: Original (blue) vs Predicted (red) Trajectories\n'
                 'Green circle = start, triangles = end points', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(project_root / 'models' / 'evaluation_grid.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: models/evaluation_grid.png")
    
    # ==========================================================================
    # Visualization 2: Error distribution
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(errors, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(errors):.4f}')
    axes[0].set_xlabel('Trajectory Error (MSE)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    sorted_indices = np.argsort(errors)
    axes[1].bar(range(len(errors)), [errors[i] for i in sorted_indices], 
                color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Sample (sorted by error)')
    axes[1].set_ylabel('Trajectory Error (MSE)')
    axes[1].set_title('Errors Ranked Best to Worst')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(project_root / 'models' / 'evaluation_errors.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: models/evaluation_errors.png")
    
    # ==========================================================================
    # Visualization 3: Best and worst predictions
    # ==========================================================================
    sorted_indices = np.argsort(errors)
    best_idx = sorted_indices[0]
    worst_idx = sorted_indices[-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, idx, title in [(axes[0], best_idx, 'Best Prediction'), 
                           (axes[1], worst_idx, 'Worst Prediction')]:
        result = results[idx]
        orig = result['original']
        pred = result['predicted']
        params = result['pred_params']
        
        ax.plot(orig[:, 0], orig[:, 1], 'b-', linewidth=3, alpha=0.8, label='Original (Target)')
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=3, alpha=0.8, label='Model Prediction')
        ax.plot(orig[0, 0], orig[0, 1], 'go', markersize=15, zorder=5, label='Start')
        ax.plot(orig[-1, 0], orig[-1, 1], 'b^', markersize=12, zorder=5, label='Target End')
        ax.plot(pred[-1, 0], pred[-1, 1], 'r^', markersize=12, zorder=5, label='Predicted End')
        
        # Draw wind field region
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (params['center'][0] - params['size'][0], 
             params['center'][1] - params['size'][1]),
            2 * params['size'][0], 2 * params['size'][1],
            fill=False, edgecolor='orange', linewidth=2, linestyle=':', 
            label='Predicted Wind Region'
        )
        ax.add_patch(rect)
        
        # Draw wind direction arrow
        ax.arrow(float(params['center'][0]), float(params['center'][1]),
                 float(params['direction'][0]) * 0.5, float(params['direction'][1]) * 0.5,
                 head_width=0.15, head_length=0.1, fc='orange', ec='orange', linewidth=2)
        
        ax.set_title(f'{title}\nError: {result["error"]:.6f}\n'
                     f'Wind strength: {params["strength"]:.2f}', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Detailed View: Best vs Worst Model Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(project_root / 'models' / 'evaluation_best_worst.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: models/evaluation_best_worst.png")
    
    # ==========================================================================
    # Visualization 4: Training history
    # ==========================================================================
    history = checkpoint['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], 'b-', linewidth=1.5, alpha=0.7, label='Train')
    axes[0].plot(history['val_loss'], 'r-', linewidth=1.5, alpha=0.7, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Last 50%
    mid = len(history['train_loss']) // 2
    axes[1].plot(range(mid, len(history['train_loss'])), 
                 history['train_loss'][mid:], 'b-', linewidth=2, label='Train')
    axes[1].plot(range(mid, len(history['val_loss'])), 
                 history['val_loss'][mid:], 'r-', linewidth=2, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training History (Last 50%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(project_root / 'models' / 'evaluation_history.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: models/evaluation_history.png")
    
    print("\n✅ Evaluation complete! Check the models/ folder for visualizations.")
    
    # Show plots
    plt.show()


if __name__ == '__main__':
    main()

