"""
TrajectoryForge Physics Engine

A JAX-based differentiable 2D physics simulator for force field optimization.
"""

from physics.state import PhysicsState, SimulationConfig
from physics.fields import (
    ForceField,
    WindField,
    VortexField,
    PointForce,
    GravityWell,
    DragZone,
)
from physics.simulator import (
    simulate_trajectory,
    simulate_positions_only,
    compute_total_force,
    trajectory_loss,
    trajectory_loss_same_length,
)
from physics.utils import (
    interpolate_trajectory,
    resample_trajectory,
    compute_trajectory_length,
)
from physics.optimization import (
    find_optimal_wind,
    find_optimal_vortex,
    optimize_adam,
    optimize_gradient_descent,
    make_trajectory_loss_fn,
    make_endpoint_loss_fn,
    OptimizationResult,
)

__all__ = [
    # State
    "PhysicsState",
    "SimulationConfig",
    # Fields
    "ForceField",
    "WindField",
    "VortexField",
    "PointForce",
    "GravityWell",
    "DragZone",
    # Simulator
    "simulate_trajectory",
    "simulate_positions_only",
    "compute_total_force",
    "trajectory_loss",
    "trajectory_loss_same_length",
    # Utils
    "interpolate_trajectory",
    "resample_trajectory",
    "compute_trajectory_length",
    # Optimization
    "find_optimal_wind",
    "find_optimal_vortex",
    "optimize_adam",
    "optimize_gradient_descent",
    "make_trajectory_loss_fn",
    "make_endpoint_loss_fn",
    "OptimizationResult",
]

