# 🎯 TrajectoryForge

**A Physics Puzzle Game with Learned AI Opponent**

TrajectoryForge is an interactive 2D physics puzzle game where players compete against an AI to configure force fields that guide a ball along a target trajectory. The core challenge is an *inverse problem*: given a desired path, find the optimal placement and parameters of force fields that minimize the error between the simulated trajectory and the target path.

## 🚀 Quick Start (AWS/Cloud)

### One-Command Setup

```bash
# Clone and setup
git clone https://github.com/ThomasKidane/trajectoryforge.git
cd trajectoryforge
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Train the Model

```bash
# Quick test (5-10 min, CPU ok)
./scripts/run_training.sh --quick

# Full training (GPU recommended, 20-30 min)
./scripts/run_training.sh --full

# Custom training
source venv/bin/activate
python scripts/train.py --model large --epochs 200 --samples 2000
```

### AWS Instance Recommendations

| Instance Type | GPU | Cost/hr | Training Time |
|---------------|-----|---------|---------------|
| `t3.large` | ❌ CPU | ~$0.08 | 2-4 hours |
| `g4dn.xlarge` | ✅ T4 | ~$0.50 | 15-30 min |
| `p3.2xlarge` | ✅ V100 | ~$3.00 | 5-10 min |

**Recommended**: `g4dn.xlarge` with Deep Learning AMI (Ubuntu)

## 🎮 Game Mechanics

### Setup
- Player sees a 2D canvas with a target trajectory (curved path)
- A ball starts at a fixed position with initial velocity
- Player places/configures force fields from a toolkit:
  - **Uniform Wind**: Constant directional force in a rectangular region
  - **Vortex/Tornado**: Tangential force, strength scales with distance from center
  - **Point Attractor/Repeller**: Inverse-square force (like electrostatics)
  - **Gravity Well**: Localized gravitational field
  - **Drag Zone**: Velocity-dependent damping region

### Gameplay Loop
1. Player configures force fields (position, strength, radius, direction)
2. Player clicks "Simulate" - ball is released and physics runs
3. Score = L2 norm of (actual_trajectory - target_trajectory)
4. Compare against AI opponent's score
5. Lower score wins!

## 🏗️ Project Structure

```
trajectory-forge/
├── physics/          # JAX-based differentiable physics engine
├── model/            # Neural network for inverse problem
├── data/             # Data generation and management
├── notebooks/        # Educational Jupyter notebooks
├── game/             # React web game
├── scripts/          # Utility scripts
└── experiments/      # Experiment configs and logs
```

## 🚀 Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+ (for web game)
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/ThomasKidane/trajectoryforge.git
cd trajectoryforge

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify installation
python scripts/verify_setup.py
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter lab

# Navigate to notebooks/ and start with 01_physics_fundamentals.ipynb
```

## 📚 Educational Notebooks

The project includes a series of educational notebooks that teach the concepts progressively:

1. **01_physics_fundamentals.ipynb** - Basic 2D physics and force fields
2. **02_differentiable_simulation.ipynb** - JAX autodiff and differentiable physics
3. **03_trajectory_optimization.ipynb** - Gradient descent on force parameters
4. **04_learning_the_inverse.ipynb** - Training neural networks for the inverse problem
5. **05_scaling_up.ipynb** - Advanced architectures and training techniques
6. **06_deployment.ipynb** - Exporting to ONNX and browser integration

## 🧠 The AI Opponent

The AI uses a neural network trained to solve the inverse problem:
- **Input**: Target trajectory (sequence of 2D points)
- **Output**: Force field configuration (positions, strengths, types)
- **Training**: End-to-end through differentiable physics simulation

Multiple difficulty levels are available:
- **Easy**: AI uses fewer optimization steps, more noise
- **Medium**: Standard AI performance
- **Hard**: Fully optimized AI solutions

## 🔬 Technical Highlights

### Differentiable Physics
- Built with JAX for automatic differentiation
- Semi-implicit Euler integration
- Smooth/soft boundaries for gradient flow
- Supports batched simulation for parallel training

### Neural Network Architecture
- Transformer encoder for variable-length trajectories
- MLP decoder with fixed max fields + validity mask
- End-to-end training through physics simulation
- Curriculum learning from simple to complex paths

### Web Game
- React + Canvas2D/Three.js
- ONNX.js for in-browser model inference
- Web Workers for non-blocking AI computation
- Real-time trajectory preview

## 📖 Resources

- [Physics-based Deep Learning](https://physicsbaseddeeplearning.org/)
- [DiffTaichi Paper](https://arxiv.org/abs/1910.00935)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Amortized Optimization Tutorial](https://arxiv.org/abs/2202.00665)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The JAX team for their incredible autodiff framework
- The physics-based deep learning community
- All contributors and testers

---

*Built with ❤️ for physics enthusiasts and game developers*

