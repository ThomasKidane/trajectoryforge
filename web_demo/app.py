#!/usr/bin/env python3
"""
TrajectoryForge Web Demo
Draw a trajectory and see the AI's prediction!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)

# =============================================================================
# Model Architecture (must match training)
# =============================================================================

class TrajectoryEncoder(nn.Module):
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
    force = direction * strength.unsqueeze(-1) * indicator
    return force


def compute_vortex_force(pos, center, radius, strength):
    diff = pos - center
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


def simulate_trajectory(init_pos, init_vel, slots, num_steps=100, dt=0.02):
    """Simulate trajectory with predicted force fields."""
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
            
            # Repeller force (type index 4)
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
# Global Model
# =============================================================================

model = None
FIELD_NAMES = ['none', 'wind', 'vortex', 'attractor', 'repeller']


def load_model():
    global model
    model_path = project_root / 'models' / 'multifield_model_final.pt'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return False
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    num_slots = checkpoint.get('num_slots', 6)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    latent_dim = checkpoint.get('latent_dim', 128)
    
    model = MultiFieldModel(hidden_dim=hidden_dim, latent_dim=latent_dim, num_slots=num_slots)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded! (epoch {checkpoint.get('epoch', 'unknown')})")
    return True


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def blog():
    return render_template('blog.html')


@app.route('/demo')
def demo():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    trajectory = data.get('trajectory', [])
    
    if len(trajectory) < 5:
        return jsonify({'error': 'Need at least 5 points'}), 400
    
    # Convert to tensor
    traj_np = np.array(trajectory, dtype=np.float32)
    
    # Resample to 100 points for the model
    if len(traj_np) != 100:
        indices = np.linspace(0, len(traj_np) - 1, 100).astype(int)
        traj_np = traj_np[indices]
    
    traj_tensor = torch.tensor(traj_np).unsqueeze(0)  # (1, 100, 2)
    
    # Get initial conditions from trajectory
    init_pos = traj_tensor[:, 0, :]  # (1, 2)
    
    # Estimate initial velocity from first few points
    if len(trajectory) > 1:
        vel = (traj_tensor[:, 1, :] - traj_tensor[:, 0, :]) / 0.02
    else:
        vel = torch.zeros(1, 2)
    
    # Run model
    with torch.no_grad():
        slots = model(traj_tensor)
        
        # Simulate with predicted fields
        pred_traj = simulate_trajectory(init_pos, vel, slots, num_steps=100, dt=0.02)
    
    # Convert results to JSON-serializable format
    pred_points = pred_traj[0].numpy().tolist()
    
    # Extract field info - return ALL slots
    fields = []
    for i, slot in enumerate(slots):
        type_probs = slot['type_probs'][0].numpy()
        type_idx = type_probs.argmax()
        type_name = FIELD_NAMES[type_idx]
        
        field_info = {
            'slot': i + 1,
            'type': type_name,
            'type_probabilities': {
                'none': float(type_probs[0]),
                'wind': float(type_probs[1]),
                'vortex': float(type_probs[2]),
                'attractor': float(type_probs[3]),
                'repeller': float(type_probs[4]),
            },
            'center': slot['center'][0].numpy().tolist(),
            'wind_params': {
                'size': slot['wind_size'][0].numpy().tolist(),
                'direction': slot['wind_direction'][0].numpy().tolist(),
                'strength': float(slot['wind_strength'][0].item()),
            },
            'vortex_params': {
                'radius': float(slot['vortex_radius'][0].item()),
                'strength': float(slot['vortex_strength'][0].item()),
            },
            'point_params': {
                'radius': float(slot['point_radius'][0].item()),
                'strength': float(slot['point_strength'][0].item()),
            },
        }
        
        # Add convenience fields for the predicted type
        if type_name == 'wind':
            field_info['size'] = field_info['wind_params']['size']
            field_info['direction'] = field_info['wind_params']['direction']
            field_info['strength'] = field_info['wind_params']['strength']
        elif type_name == 'vortex':
            field_info['radius'] = field_info['vortex_params']['radius']
            field_info['strength'] = field_info['vortex_params']['strength']
        elif type_name in ['attractor', 'repeller']:
            field_info['radius'] = field_info['point_params']['radius']
            field_info['strength'] = field_info['point_params']['strength']
        
        fields.append(field_info)
    
    return jsonify({
        'predicted_trajectory': pred_points,
        'fields': fields,
    })


if __name__ == '__main__':
    if load_model():
        print("\n🚀 Starting TrajectoryForge Demo Server...")
        print("   Open http://localhost:5000 in your browser\n")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load model!")

