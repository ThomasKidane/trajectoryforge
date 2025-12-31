#!/usr/bin/env python3
"""
Verify TrajectoryForge Setup

Run this script to verify that all components are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path (parent of scripts/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"  ✅ JAX {jax.__version__}")
    except ImportError as e:
        print(f"  ❌ JAX: {e}")
        return False
    
    try:
        from physics.state import PhysicsState, SimulationConfig
        from physics.fields import WindField, VortexField, PointForce
        from physics.simulator import simulate_trajectory, trajectory_loss_same_length
        from physics.utils import generate_arc_trajectory
        print("  ✅ Physics module")
    except ImportError as e:
        print(f"  ❌ Physics module: {e}")
        return False
    
    try:
        from model.encoder import TrajectoryEncoder
        from model.decoder import ForceFieldDecoder
        from model.model import InverseModel
        print("  ✅ Model module")
    except ImportError as e:
        print(f"  ❌ Model module: {e}")
        return False
    
    try:
        from data.generator import generate_random_fields, generate_dataset_sample
        from data.augmentation import augment_trajectory
        print("  ✅ Data module")
    except ImportError as e:
        print(f"  ❌ Data module: {e}")
        return False
    
    return True


def test_physics():
    """Test basic physics simulation."""
    print("\nTesting physics simulation...")
    
    import jax.numpy as jnp
    from physics.state import PhysicsState, SimulationConfig
    from physics.fields import WindField
    from physics.simulator import simulate_trajectory
    
    # Create a simple wind field
    wind = WindField.create(
        center=(1.5, 0.5),
        size=(2.0, 2.0),
        direction=(0.0, 1.0),
        strength=5.0
    )
    
    # Create initial state
    state = PhysicsState.create(
        position=(0.0, 0.0),
        velocity=(1.5, 0.0),
        time=0.0
    )
    
    # Run simulation
    config = SimulationConfig(dt=0.01, num_steps=100)
    trajectory = simulate_trajectory(state, [wind], config)
    
    print(f"  ✅ Simulated {trajectory.num_points} steps")
    print(f"  ✅ Final position: ({trajectory.positions[-1, 0]:.2f}, {trajectory.positions[-1, 1]:.2f})")
    
    return True


def test_differentiability():
    """Test that gradients can be computed."""
    print("\nTesting differentiability...")
    
    import jax
    import jax.numpy as jnp
    from jax import grad
    from physics.state import PhysicsState, SimulationConfig
    from physics.fields import WindField
    from physics.simulator import simulate_positions_only
    
    def loss_fn(wind_strength):
        wind = WindField(
            center=jnp.array([1.5, 0.5]),
            size=jnp.array([2.0, 2.0]),
            direction=jnp.array([0.0, 1.0]),
            strength=wind_strength,
            softness=0.1
        )
        state = PhysicsState(
            position=jnp.array([0.0, 0.0]),
            velocity=jnp.array([1.5, 0.0]),
            time=0.0
        )
        config = SimulationConfig(dt=0.01, num_steps=100)
        positions = simulate_positions_only(state, [wind], config)
        return jnp.sum(positions[-1] ** 2)
    
    # Compute gradient
    grad_fn = grad(loss_fn)
    gradient = grad_fn(5.0)
    
    print(f"  ✅ Gradient computed: {gradient:.4f}")
    
    return True


def test_data_generation():
    """Test data generation."""
    print("\nTesting data generation...")
    
    from jax import random
    from data.generator import generate_dataset_sample
    
    key = random.PRNGKey(42)
    sample = generate_dataset_sample(key, num_wind=1, num_vortex=1)
    
    print(f"  ✅ Generated trajectory with {sample.trajectory.shape[0]} points")
    print(f"  ✅ Initial position: ({sample.initial_position[0]:.2f}, {sample.initial_position[1]:.2f})")
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("TrajectoryForge Setup Verification")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_physics():
        all_passed = False
    
    if not test_differentiability():
        all_passed = False
    
    if not test_data_generation():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! TrajectoryForge is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

