#!/usr/bin/env python3
"""
Evaluate Multi-Field Model

Visualizes predictions from the multi-field model, showing:
- Original vs predicted trajectories
- Which force fields the model predicted
- Field type breakdown
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from scipy.interpolate import interp1d

os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import random
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS

# Import model classes from training script
from scripts.train_multifield import (
    MultiFieldModel, simulate_trajectory_multifield
)


FIELD_COLORS = {
    'none': '#666666',
    'wind': '#40e0d0',
    'vortex': '#8a2be2', 
    'attractor': '#32cd32',
    'repeller': '#ff4500',
}

FIELD_NAMES = ['none', 'wind', 'vortex', 'attractor', 'repeller']


def load_model(model_path, device):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    model = MultiFieldModel(
        hidden_dim=args['hidden_dim'],
        latent_dim=args['latent_dim'],
        num_slots=args['num_slots']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Num slots: {args['num_slots']}")
    
    return model, args


def generate_test_samples(num_samples=20, seed=123):
    """Generate test samples from all difficulties."""
    key = random.PRNGKey(seed)
    samples = []
    
    difficulties = ['easy', 'medium', 'hard', 'expert']
    per_diff = num_samples // len(difficulties)
    
    for diff in difficulties:
        level = DIFFICULTY_LEVELS[diff]
        for _ in range(per_diff):
            key, subkey = random.split(key)
            try:
                sample = generate_sample_at_difficulty(subkey, level)
                sample['difficulty'] = diff
                samples.append(sample)
            except:
                continue
    
    return samples


def resample_trajectory(traj, target_length):
    """Resample trajectory to target length."""
    current_len = len(traj)
    if current_len == target_length:
        return traj
    
    old_idx = np.linspace(0, 1, current_len)
    new_idx = np.linspace(0, 1, target_length)
    
    interp_x = interp1d(old_idx, traj[:, 0], kind='linear')
    interp_y = interp1d(old_idx, traj[:, 1], kind='linear')
    
    new_traj = np.stack([interp_x(new_idx), interp_y(new_idx)], axis=1)
    return new_traj


def predict_and_simulate(model, sample, device, target_length=100):
    """Get model prediction and simulate trajectory."""
    traj = np.array(sample['trajectory'])
    traj = resample_trajectory(traj, target_length)
    
    traj_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(device)
    init_pos = torch.tensor(np.array(sample['initial_position']), dtype=torch.float32).unsqueeze(0).to(device)
    init_vel = torch.tensor(np.array(sample['initial_velocity']), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        slots = model(traj_tensor)
        pred_traj = simulate_trajectory_multifield(
            init_pos, init_vel, slots, dt=0.01, num_steps=target_length
        )
    
    # Extract slot info
    slot_info = []
    for slot in slots:
        type_probs = slot['type_probs'][0].cpu().numpy()
        type_idx = np.argmax(type_probs)
        
        info = {
            'type': FIELD_NAMES[type_idx],
            'type_prob': type_probs[type_idx],
            'type_probs': type_probs,
            'center': slot['center'][0].cpu().numpy(),
        }
        
        if type_idx == 1:  # wind
            info['size'] = slot['wind_size'][0].cpu().numpy()
            info['direction'] = slot['wind_direction'][0].cpu().numpy()
            info['strength'] = slot['wind_strength'][0].cpu().item()
        elif type_idx == 2:  # vortex
            info['radius'] = slot['vortex_radius'][0].cpu().item()
            info['strength'] = slot['vortex_strength'][0].cpu().item()
        elif type_idx in [3, 4]:  # attractor/repeller
            info['radius'] = slot['point_radius'][0].cpu().item()
            info['strength'] = slot['point_strength'][0].cpu().item()
        
        slot_info.append(info)
    
    return {
        'original': traj,
        'predicted': pred_traj[0].cpu().numpy(),
        'slots': slot_info,
    }


def draw_field(ax, slot_info, alpha=0.3):
    """Draw a force field on the axes."""
    field_type = slot_info['type']
    if field_type == 'none':
        return
    
    center = slot_info['center']
    color = FIELD_COLORS[field_type]
    
    if field_type == 'wind':
        size = slot_info['size']
        rect = Rectangle(
            (center[0] - size[0], center[1] - size[1]),
            size[0] * 2, size[1] * 2,
            fill=True, facecolor=color, edgecolor=color,
            alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        # Draw direction arrow
        direction = slot_info['direction']
        ax.arrow(center[0], center[1], 
                direction[0] * 0.5, direction[1] * 0.5,
                head_width=0.15, head_length=0.1, fc=color, ec=color)
        
    elif field_type == 'vortex':
        radius = slot_info['radius']
        circle = Circle(center, radius, fill=True, facecolor=color, 
                       edgecolor=color, alpha=alpha, linewidth=2)
        ax.add_patch(circle)
        # Draw spiral indicator
        ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        
    elif field_type in ['attractor', 'repeller']:
        radius = slot_info['radius']
        circle = Circle(center, radius, fill=True, facecolor=color,
                       edgecolor=color, alpha=alpha, linewidth=2)
        ax.add_patch(circle)
        ax.plot(center[0], center[1], '+' if field_type == 'attractor' else 'x',
               color=color, markersize=12, markeredgewidth=3)


def plot_evaluation_grid(samples, results, save_path='models/multifield_eval.png'):
    """Create grid visualization."""
    n = len(samples)
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (sample, result) in enumerate(zip(samples, results)):
        ax = axes[i]
        
        orig = result['original']
        pred = result['predicted']
        
        # Draw force fields
        for slot in result['slots']:
            if slot['type'] != 'none' and slot['type_prob'] > 0.3:
                draw_field(ax, slot, alpha=0.25)
        
        # Draw trajectories
        ax.plot(orig[:, 0], orig[:, 1], 'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.7)
        
        # Start and end points
        ax.plot(orig[0, 0], orig[0, 1], 'go', markersize=10, label='Start')
        ax.plot(orig[-1, 0], orig[-1, 1], 'r*', markersize=12, label='End')
        
        # Calculate error
        error = np.mean(np.linalg.norm(orig - pred, axis=1))
        
        # Count active fields
        active_fields = [s['type'] for s in result['slots'] if s['type'] != 'none' and s['type_prob'] > 0.3]
        field_str = ', '.join(active_fields) if active_fields else 'none'
        
        ax.set_title(f"{sample['difficulty'].upper()}\nError: {error:.3f}\nFields: {field_str}", fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Hide empty subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_field_usage(results, save_path='models/multifield_field_usage.png'):
    """Plot field type usage statistics."""
    type_counts = {name: 0 for name in FIELD_NAMES}
    
    for result in results:
        for slot in result['slots']:
            if slot['type_prob'] > 0.3:
                type_counts[slot['type']] += 1
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(type_counts.keys())
    counts = list(type_counts.values())
    colors = [FIELD_COLORS[n] for n in names]
    
    bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Field Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Predicted Field Type Usage', fontsize=14)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/multifield_model.pt')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model, model_args = load_model(args.model, device)
    
    # Generate test samples
    print(f"\nGenerating {args.num_samples} test samples...")
    samples = generate_test_samples(args.num_samples, args.seed)
    print(f"Generated {len(samples)} samples")
    
    # Evaluate
    print("\nEvaluating...")
    results = []
    for sample in tqdm(samples):
        result = predict_and_simulate(model, sample, device)
        results.append(result)
    
    # Calculate statistics
    errors = []
    for sample, result in zip(samples, results):
        orig = result['original']
        pred = result['predicted']
        error = np.mean(np.linalg.norm(orig - pred, axis=1))
        errors.append(error)
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Mean error: {np.mean(errors):.4f}")
    print(f"Std error: {np.std(errors):.4f}")
    print(f"Min error: {np.min(errors):.4f}")
    print(f"Max error: {np.max(errors):.4f}")
    
    # By difficulty
    for diff in ['easy', 'medium', 'hard', 'expert']:
        diff_errors = [e for s, e in zip(samples, errors) if s['difficulty'] == diff]
        if diff_errors:
            print(f"  {diff}: {np.mean(diff_errors):.4f} ± {np.std(diff_errors):.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_evaluation_grid(samples, results)
    plot_field_usage(results)
    
    print("\nDone!")


if __name__ == '__main__':
    from tqdm import tqdm
    main()

