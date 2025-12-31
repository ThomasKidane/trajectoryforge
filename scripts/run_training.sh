#!/bin/bash
# Quick start script for training on AWS
# Usage: ./scripts/run_training.sh [--quick|--full]

set -e

echo "=============================================="
echo "TrajectoryForge Training"
echo "=============================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "   Run ./scripts/setup.sh first"
    exit 1
fi

# Default to full training if on GPU, quick otherwise
MODE="--full"
if [ "$1" != "" ]; then
    MODE="$1"
fi

# Check for GPU
if python -c "import jax; print('gpu' in str(jax.devices()).lower())" | grep -q "True"; then
    echo "🚀 GPU detected - running full training"
else
    echo "💻 CPU only - consider using --quick for faster results"
fi

# Run training
echo ""
echo "Starting training with mode: $MODE"
echo ""

python scripts/train.py $MODE --output models/trained_model.pkl

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: models/trained_model.pkl"
echo "=============================================="

