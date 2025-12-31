"""
Differentiable Physics Simulator

This module implements the core simulation loop using JAX for automatic
differentiation. The simulator can compute gradients of the trajectory
with respect to all force field parameters.

Key Features:
- Semi-implicit Euler integration for stability
- Soft boundary handling for smooth gradients
- Batched simulation support
- Full differentiability through jax.lax.scan
"""

from __future__ import annotations
from typing import Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array, lax

from physics.state import PhysicsState, SimulationConfig, TrajectoryData
from physics.fields import (
    AnyField,
    WindField,
    VortexField,
    PointForce,
    GravityWell,
    DragZone,
)


# =============================================================================
# Force Computation
# =============================================================================

def compute_single_field_force(
    field: AnyField,
    position: Array,
    velocity: Array
) -> Array:
    """
    Compute force from a single field.
    
    This function handles the type dispatch for different field types
    in a JAX-compatible way.
    """
    return field.compute_force(position, velocity)


def compute_total_force(
    position: Array,
    velocity: Array,
    fields: Sequence[AnyField],
    config: SimulationConfig
) -> Array:
    """
    Compute the total force on a particle from all fields and global effects.
    
    Args:
        position: Current position, shape (2,)
        velocity: Current velocity, shape (2,)
        fields: List of force fields
        config: Simulation configuration
        
    Returns:
        Total force vector, shape (2,)
    """
    # Start with global gravity
    total_force = config.gravity_array
    
    # Add global drag
    if config.drag_coefficient > 0:
        total_force = total_force - config.drag_coefficient * velocity
    
    # Sum forces from all fields
    for field in fields:
        total_force = total_force + field.compute_force(position, velocity)
    
    # Add soft boundary forces
    total_force = total_force + compute_boundary_force(position, config)
    
    return total_force


def compute_boundary_force(position: Array, config: SimulationConfig) -> Array:
    """
    Compute soft boundary repulsion force.
    
    Uses a smooth potential that increases exponentially as the particle
    approaches the boundary. This ensures gradients flow smoothly.
    
    Args:
        position: Current position, shape (2,)
        config: Simulation configuration
        
    Returns:
        Boundary force vector, shape (2,)
    """
    bounds = config.bounds_array
    softness = config.boundary_softness
    strength = config.boundary_strength
    
    force = jnp.zeros(2)
    
    # X boundaries
    x_min, x_max = bounds[0]
    # Force pushing away from left boundary
    dist_left = position[0] - x_min
    force_left = strength * jnp.exp(-dist_left / softness)
    # Force pushing away from right boundary
    dist_right = x_max - position[0]
    force_right = -strength * jnp.exp(-dist_right / softness)
    
    force = force.at[0].set(force_left + force_right)
    
    # Y boundaries
    y_min, y_max = bounds[1]
    # Force pushing away from bottom boundary
    dist_bottom = position[1] - y_min
    force_bottom = strength * jnp.exp(-dist_bottom / softness)
    # Force pushing away from top boundary
    dist_top = y_max - position[1]
    force_top = -strength * jnp.exp(-dist_top / softness)
    
    force = force.at[1].set(force_bottom + force_top)
    
    return force


# =============================================================================
# Integration Step
# =============================================================================

def semi_implicit_euler_step(
    state: PhysicsState,
    fields: Sequence[AnyField],
    config: SimulationConfig
) -> PhysicsState:
    """
    Perform one semi-implicit Euler integration step.
    
    Semi-implicit Euler (also called symplectic Euler) updates velocity first,
    then uses the new velocity to update position. This provides better
    energy conservation than explicit Euler.
    
    v_{n+1} = v_n + (F/m) * dt
    x_{n+1} = x_n + v_{n+1} * dt
    
    Args:
        state: Current physics state
        fields: List of force fields
        config: Simulation configuration
        
    Returns:
        New physics state after one timestep
    """
    # Compute total force
    force = compute_total_force(state.position, state.velocity, fields, config)
    
    # Compute acceleration (F = ma, so a = F/m)
    acceleration = force / config.ball_mass
    
    # Update velocity first (semi-implicit)
    new_velocity = state.velocity + acceleration * config.dt
    
    # Then update position using new velocity
    new_position = state.position + new_velocity * config.dt
    
    # Update time
    new_time = state.time + config.dt
    
    return PhysicsState(
        position=new_position,
        velocity=new_velocity,
        time=new_time
    )


# =============================================================================
# Main Simulation Functions
# =============================================================================

def simulate_trajectory(
    initial_state: PhysicsState,
    fields: Sequence[AnyField],
    config: SimulationConfig | None = None
) -> TrajectoryData:
    """
    Run a full simulation and return the trajectory.
    
    This function is fully differentiable with respect to:
    - Initial state (position, velocity)
    - Field parameters (positions, strengths, etc.)
    
    Uses jax.lax.scan for efficient, differentiable looping.
    
    Args:
        initial_state: Starting physics state
        fields: List of force fields
        config: Simulation configuration (uses default if None)
        
    Returns:
        TrajectoryData containing positions, velocities, times, and forces
        
    Example:
        >>> state = PhysicsState.create(position=(0, 0), velocity=(1, 0))
        >>> fields = [WindField.create((2, 0), (2, 2), (0, 1), 5.0)]
        >>> config = SimulationConfig(num_steps=100)
        >>> traj = simulate_trajectory(state, fields, config)
        >>> traj.positions.shape
        (100, 2)
    """
    if config is None:
        config = SimulationConfig()
    
    def scan_fn(carry, _):
        """Scan function for jax.lax.scan."""
        state = carry
        
        # Compute force for recording
        force = compute_total_force(state.position, state.velocity, fields, config)
        
        # Take integration step
        new_state = semi_implicit_euler_step(state, fields, config)
        
        # Output: (position, velocity, time, force)
        output = (state.position, state.velocity, state.time, force)
        
        return new_state, output
    
    # Run simulation using scan
    _, outputs = lax.scan(
        scan_fn,
        initial_state,
        None,  # No input sequence needed
        length=config.num_steps
    )
    
    positions, velocities, times, forces = outputs
    
    return TrajectoryData(
        positions=positions,
        velocities=velocities,
        times=jnp.array(times),
        forces=forces
    )


def simulate_positions_only(
    initial_state: PhysicsState,
    fields: Sequence[AnyField],
    config: SimulationConfig | None = None
) -> Array:
    """
    Run simulation and return only positions (more memory efficient).
    
    Use this when you only need the trajectory path and want to
    minimize memory usage during training.
    
    Args:
        initial_state: Starting physics state
        fields: List of force fields
        config: Simulation configuration
        
    Returns:
        Positions array, shape (num_steps, 2)
    """
    if config is None:
        config = SimulationConfig()
    
    def scan_fn(carry, _):
        state = carry
        new_state = semi_implicit_euler_step(state, fields, config)
        return new_state, state.position
    
    _, positions = lax.scan(
        scan_fn,
        initial_state,
        None,
        length=config.num_steps
    )
    
    return positions


# =============================================================================
# Batched Simulation
# =============================================================================

@partial(jax.vmap, in_axes=(0, None, None))
def simulate_batch_initial_states(
    initial_state: PhysicsState,
    fields: Sequence[AnyField],
    config: SimulationConfig
) -> Array:
    """
    Simulate multiple initial states with the same fields (batched).
    
    Args:
        initial_state: Batched initial states (vmapped over first axis)
        fields: Shared force fields
        config: Simulation configuration
        
    Returns:
        Batched positions, shape (batch_size, num_steps, 2)
    """
    return simulate_positions_only(initial_state, fields, config)


# =============================================================================
# Loss Functions
# =============================================================================

def trajectory_loss(
    predicted_traj: Array,
    target_traj: Array,
    reduction: str = "mean"
) -> float:
    """
    Compute L2 loss between predicted and target trajectories.
    
    Handles trajectories of different lengths by interpolating the
    shorter one to match the longer one.
    
    Args:
        predicted_traj: Predicted positions, shape (N, 2)
        target_traj: Target positions, shape (M, 2)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Scalar loss value (or array if reduction='none')
    """
    # Get lengths
    n_pred = predicted_traj.shape[0]
    n_target = target_traj.shape[0]
    
    # If lengths match, compute directly
    # Otherwise, we need to interpolate
    # For simplicity, we'll resample both to a common length
    
    # Use the longer length as reference
    n_common = jnp.maximum(n_pred, n_target)
    
    # Resample both trajectories
    pred_resampled = resample_trajectory_to_length(predicted_traj, n_common)
    target_resampled = resample_trajectory_to_length(target_traj, n_common)
    
    # Compute pointwise squared distances
    squared_dists = jnp.sum((pred_resampled - target_resampled) ** 2, axis=1)
    
    if reduction == "mean":
        return jnp.mean(squared_dists)
    elif reduction == "sum":
        return jnp.sum(squared_dists)
    else:
        return squared_dists


def trajectory_loss_same_length(
    predicted_traj: Array,
    target_traj: Array
) -> float:
    """
    Compute L2 loss assuming trajectories have the same length.
    
    More efficient than trajectory_loss when lengths are known to match.
    
    Args:
        predicted_traj: Predicted positions, shape (N, 2)
        target_traj: Target positions, shape (N, 2)
        
    Returns:
        Mean squared distance
    """
    squared_dists = jnp.sum((predicted_traj - target_traj) ** 2, axis=1)
    return jnp.mean(squared_dists)


def endpoint_loss(predicted_traj: Array, target_endpoint: Array) -> float:
    """
    Compute loss based only on final position.
    
    Useful for simpler optimization targets.
    
    Args:
        predicted_traj: Predicted positions, shape (N, 2)
        target_endpoint: Target final position, shape (2,)
        
    Returns:
        Squared distance to target endpoint
    """
    final_pos = predicted_traj[-1]
    return jnp.sum((final_pos - target_endpoint) ** 2)


def waypoint_loss(
    predicted_traj: Array,
    waypoints: Array,
    waypoint_times: Array,
    dt: float
) -> float:
    """
    Compute loss based on passing through specific waypoints at specific times.
    
    Args:
        predicted_traj: Predicted positions, shape (N, 2)
        waypoints: Target waypoint positions, shape (K, 2)
        waypoint_times: Times when waypoints should be reached, shape (K,)
        dt: Simulation timestep
        
    Returns:
        Sum of squared distances to waypoints
    """
    # Convert times to indices
    indices = (waypoint_times / dt).astype(jnp.int32)
    indices = jnp.clip(indices, 0, predicted_traj.shape[0] - 1)
    
    # Get predicted positions at waypoint times
    predicted_at_waypoints = predicted_traj[indices]
    
    # Compute squared distances
    squared_dists = jnp.sum((predicted_at_waypoints - waypoints) ** 2, axis=1)
    
    return jnp.sum(squared_dists)


def smoothness_loss(trajectory: Array) -> float:
    """
    Compute smoothness penalty based on acceleration (second derivative).
    
    Encourages smooth trajectories without sudden direction changes.
    
    Args:
        trajectory: Positions, shape (N, 2)
        
    Returns:
        Mean squared acceleration
    """
    # First derivative (velocity proxy)
    vel = jnp.diff(trajectory, axis=0)
    # Second derivative (acceleration proxy)
    acc = jnp.diff(vel, axis=0)
    
    return jnp.mean(jnp.sum(acc ** 2, axis=1))


# =============================================================================
# Trajectory Resampling (for loss computation)
# =============================================================================

def resample_trajectory_to_length(trajectory: Array, target_length: int) -> Array:
    """
    Resample a trajectory to a specific number of points.
    
    Uses linear interpolation for smooth, differentiable resampling.
    
    Args:
        trajectory: Input positions, shape (N, 2)
        target_length: Desired number of points
        
    Returns:
        Resampled positions, shape (target_length, 2)
    """
    n_points = trajectory.shape[0]
    
    # Parameter along original trajectory [0, 1]
    t_original = jnp.linspace(0, 1, n_points)
    
    # Parameter for resampled trajectory
    t_target = jnp.linspace(0, 1, target_length)
    
    # Interpolate each dimension
    x_resampled = jnp.interp(t_target, t_original, trajectory[:, 0])
    y_resampled = jnp.interp(t_target, t_original, trajectory[:, 1])
    
    return jnp.stack([x_resampled, y_resampled], axis=1)


# =============================================================================
# Gradient Computation Utilities
# =============================================================================

def make_loss_fn(
    initial_state: PhysicsState,
    target_trajectory: Array,
    config: SimulationConfig
):
    """
    Create a loss function that takes field parameters and returns loss.
    
    This is useful for optimization - the returned function can be
    differentiated with jax.grad.
    
    Args:
        initial_state: Fixed initial state
        target_trajectory: Target trajectory to match
        config: Simulation configuration
        
    Returns:
        Function: field_params -> loss
    """
    def loss_fn(field_params_list: list[dict]) -> float:
        """Compute loss for given field parameters."""
        from physics.fields import params_to_fields
        
        fields = params_to_fields(field_params_list)
        predicted = simulate_positions_only(initial_state, fields, config)
        return trajectory_loss_same_length(predicted, target_trajectory)
    
    return loss_fn


# =============================================================================
# Visualization Helpers
# =============================================================================

def simulate_with_substeps(
    initial_state: PhysicsState,
    fields: Sequence[AnyField],
    config: SimulationConfig,
    substeps: int = 10
) -> TrajectoryData:
    """
    Simulate with sub-step recording for smoother visualization.
    
    Records position at every substep for smoother trajectory plots,
    but physics is computed at the original timestep.
    
    Args:
        initial_state: Starting state
        fields: Force fields
        config: Simulation configuration
        substeps: Number of substeps to record per physics step
        
    Returns:
        TrajectoryData with substeps * num_steps points
    """
    # Create config with smaller timestep
    fine_config = SimulationConfig(
        dt=config.dt / substeps,
        num_steps=config.num_steps * substeps,
        ball_mass=config.ball_mass,
        ball_radius=config.ball_radius,
        bounds=config.bounds,
        boundary_softness=config.boundary_softness,
        boundary_strength=config.boundary_strength,
        global_gravity=config.global_gravity,
        drag_coefficient=config.drag_coefficient
    )
    
    return simulate_trajectory(initial_state, fields, fine_config)

