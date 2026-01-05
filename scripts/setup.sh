#!/bin/bash
# TrajectoryForge Setup Script
# Run this on a fresh AWS instance (Ubuntu 22.04 recommended)

set -e  # Exit on error

echo "=============================================="
echo "TrajectoryForge Setup"
echo "=============================================="

# Update system
echo ""
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and dependencies
echo ""
echo "🐍 Installing Python 3.10..."
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install JAX (CPU version by default)
echo ""
echo "📥 Installing JAX..."

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU detected! Installing JAX with CUDA support..."
    pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "   No GPU detected. Installing CPU version..."
    pip install --upgrade jax jaxlib
fi

# Install other requirements
echo ""
echo "📥 Installing Python packages..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "
import jax
import jax.numpy as jnp
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')

# Quick test
from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField
from physics.simulator import simulate_positions_only
from model.architectures import create_model_from_config

model = create_model_from_config('tiny')
print('✅ All imports working!')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To train the model, run:"
echo "  source venv/bin/activate"
echo "  python scripts/train.py --full"
echo ""
echo "Options:"
echo "  --quick   : Quick test (5-10 min)"
echo "  --full    : Full training (GPU recommended)"
echo "  --epochs N: Custom number of epochs"
echo ""




