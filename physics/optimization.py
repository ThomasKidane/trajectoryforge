"""
Trajectory Optimization

This module provides gradient-based optimization for finding force field
configurations that produce desired trajectories.

Key Features:
- Multiple loss functions (trajectory, endpoint, waypoint)
- Gradient descent with various optimizers
- Visualization of optimization progress
- Support for constraints and regularization
"""

from __future__ import annotations
from typing import NamedTuple, Callable, Any
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
from tqdm import tqdm

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField, VortexField, PointForce, AnyField
from physics.simulator import simulate_positions_only, trajectory_loss_same_length


# =============================================================================
# Optimization State and History
# =============================================================================

class OptimizationResult(NamedTuple):
    """Result of an optimization run."""
    final_params: dict
    final_loss: float
    history: dict
    converged: bool
    num_iterations: int


@dataclass
class OptimizationHistory:
    """Tracks optimization progress."""
    losses: list[float] = field(default_factory=list)
    params: list[dict] = field(default_factory=list)
    gradients: list[dict] = field(default_factory=list)
    trajectories: list[jnp.ndarray] = field(default_factory=list)
    
    def record(self, loss: float, params: dict, gradient: dict = None, trajectory: jnp.ndarray = None):
        """Record a single optimization step."""
        self.losses.append(float(loss))
        self.params.append(params)
        if gradient is not None:
            self.gradients.append(gradient)
        if trajectory is not None:
            self.trajectories.append(trajectory)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'losses': self.losses,
            'params': self.params,
            'num_steps': len(self.losses),
        }


# =============================================================================
# Parameterization Helpers
# =============================================================================

def params_to_wind_field(params: dict) -> WindField:
    """Convert parameter dict to WindField."""
    direction = params['direction']
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    return WindField(
        center=params['center'],
        size=params['size'],
        direction=direction,
        strength=params['strength'],
        softness=params.get('softness', 0.1),
    )


def params_to_vortex_field(params: dict) -> VortexField:
    """Convert parameter dict to VortexField."""
    return VortexField(
        center=params['center'],
        strength=params['strength'],
        radius=params['radius'],
        falloff=params.get('falloff', 'gaussian'),
        softness=params.get('softness', 0.1),
    )


def params_to_point_force(params: dict) -> PointForce:
    """Convert parameter dict to PointForce."""
    return PointForce(
        center=params['center'],
        strength=params['strength'],
        falloff_power=params.get('falloff_power', 2.0),
        max_distance=params.get('max_distance', 10.0),
        softness=params.get('softness', 0.2),
    )


def create_wind_params(
    center: tuple = (0.0, 0.0),
    size: tuple = (2.0, 2.0),
    direction: tuple = (0.0, 1.0),
    strength: float = 5.0,
) -> dict:
    """Create initial parameters for a wind field."""
    return {
        'center': jnp.array(center, dtype=jnp.float32),
        'size': jnp.array(size, dtype=jnp.float32),
        'direction': jnp.array(direction, dtype=jnp.float32),
        'strength': jnp.float32(strength),
    }


def create_vortex_params(
    center: tuple = (0.0, 0.0),
    strength: float = 5.0,
    radius: float = 1.5,
) -> dict:
    """Create initial parameters for a vortex field."""
    return {
        'center': jnp.array(center, dtype=jnp.float32),
        'strength': jnp.float32(strength),
        'radius': jnp.float32(radius),
    }


def create_point_params(
    center: tuple = (0.0, 0.0),
    strength: float = 5.0,
) -> dict:
    """Create initial parameters for a point force."""
    return {
        'center': jnp.array(center, dtype=jnp.float32),
        'strength': jnp.float32(strength),
    }


# =============================================================================
# Loss Functions
# =============================================================================

def make_trajectory_loss_fn(
    initial_state: PhysicsState,
    target_trajectory: jnp.ndarray,
    config: SimulationConfig,
    field_type: str = 'wind',
    smoothness_weight: float = 0.0,
) -> Callable:
    """
    Create a loss function for trajectory matching.
    
    Args:
        initial_state: Starting physics state
        target_trajectory: Target trajectory to match, shape (N, 2)
        config: Simulation configuration
        field_type: Type of field to optimize ('wind', 'vortex', 'point')
        smoothness_weight: Weight for trajectory smoothness regularization
        
    Returns:
        Loss function: params -> scalar loss
    """
    def loss_fn(params: dict) -> float:
        # Convert params to field
        if field_type == 'wind':
            field = params_to_wind_field(params)
        elif field_type == 'vortex':
            field = params_to_vortex_field(params)
        elif field_type == 'point':
            field = params_to_point_force(params)
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Simulate trajectory
        predicted = simulate_positions_only(initial_state, [field], config)
        
        # Main loss: trajectory matching
        loss = trajectory_loss_same_length(predicted, target_trajectory)
        
        # Optional: smoothness regularization
        if smoothness_weight > 0:
            vel = jnp.diff(predicted, axis=0)
            acc = jnp.diff(vel, axis=0)
            smoothness_loss = jnp.mean(jnp.sum(acc ** 2, axis=1))
            loss = loss + smoothness_weight * smoothness_loss
        
        return loss
    
    return loss_fn


def make_endpoint_loss_fn(
    initial_state: PhysicsState,
    target_endpoint: jnp.ndarray,
    config: SimulationConfig,
    field_type: str = 'wind',
) -> Callable:
    """
    Create a loss function for endpoint matching.
    
    Simpler than full trajectory matching - just match the final position.
    """
    def loss_fn(params: dict) -> float:
        if field_type == 'wind':
            field = params_to_wind_field(params)
        elif field_type == 'vortex':
            field = params_to_vortex_field(params)
        elif field_type == 'point':
            field = params_to_point_force(params)
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        predicted = simulate_positions_only(initial_state, [field], config)
        final_pos = predicted[-1]
        
        return jnp.sum((final_pos - target_endpoint) ** 2)
    
    return loss_fn


def make_waypoint_loss_fn(
    initial_state: PhysicsState,
    waypoints: jnp.ndarray,
    waypoint_times: jnp.ndarray,
    config: SimulationConfig,
    field_type: str = 'wind',
) -> Callable:
    """
    Create a loss function for waypoint matching.
    
    The trajectory must pass through specific points at specific times.
    """
    def loss_fn(params: dict) -> float:
        if field_type == 'wind':
            field = params_to_wind_field(params)
        elif field_type == 'vortex':
            field = params_to_vortex_field(params)
        elif field_type == 'point':
            field = params_to_point_force(params)
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        predicted = simulate_positions_only(initial_state, [field], config)
        
        # Convert times to indices
        indices = (waypoint_times / config.dt).astype(jnp.int32)
        indices = jnp.clip(indices, 0, config.num_steps - 1)
        
        # Get predicted positions at waypoint times
        predicted_at_waypoints = predicted[indices]
        
        # Loss: sum of squared distances
        return jnp.sum((predicted_at_waypoints - waypoints) ** 2)
    
    return loss_fn


def make_multi_field_loss_fn(
    initial_state: PhysicsState,
    target_trajectory: jnp.ndarray,
    config: SimulationConfig,
    field_types: list[str],
) -> Callable:
    """
    Create a loss function for optimizing multiple fields simultaneously.
    
    Args:
        initial_state: Starting physics state
        target_trajectory: Target trajectory to match
        config: Simulation configuration
        field_types: List of field types, e.g., ['wind', 'vortex']
        
    Returns:
        Loss function: params_list -> scalar loss
    """
    def loss_fn(params_list: list[dict]) -> float:
        fields = []
        for params, ftype in zip(params_list, field_types):
            if ftype == 'wind':
                fields.append(params_to_wind_field(params))
            elif ftype == 'vortex':
                fields.append(params_to_vortex_field(params))
            elif ftype == 'point':
                fields.append(params_to_point_force(params))
        
        predicted = simulate_positions_only(initial_state, fields, config)
        return trajectory_loss_same_length(predicted, target_trajectory)
    
    return loss_fn


# =============================================================================
# Optimizers
# =============================================================================

def optimize_gradient_descent(
    loss_fn: Callable,
    initial_params: dict,
    learning_rate: float = 0.1,
    num_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    verbose: bool = True,
    record_trajectory: bool = False,
    initial_state: PhysicsState = None,
    config: SimulationConfig = None,
    field_type: str = 'wind',
) -> OptimizationResult:
    """
    Optimize using simple gradient descent.
    
    Args:
        loss_fn: Loss function to minimize
        initial_params: Starting parameter values
        learning_rate: Step size for gradient descent
        num_iterations: Maximum number of iterations
        convergence_threshold: Stop if loss change is below this
        verbose: Whether to print progress
        record_trajectory: Whether to record trajectories at each step
        
    Returns:
        OptimizationResult with final params, loss, and history
    """
    params = initial_params.copy()
    history = OptimizationHistory()
    
    grad_fn = grad(loss_fn)
    
    prev_loss = float('inf')
    
    iterator = range(num_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="Optimizing")
    
    for i in iterator:
        loss = loss_fn(params)
        gradient = grad_fn(params)
        
        # Record history
        traj = None
        if record_trajectory and initial_state is not None and config is not None:
            if field_type == 'wind':
                field = params_to_wind_field(params)
            elif field_type == 'vortex':
                field = params_to_vortex_field(params)
            elif field_type == 'point':
                field = params_to_point_force(params)
            traj = simulate_positions_only(initial_state, [field], config)
        
        history.record(loss, params.copy(), gradient, traj)
        
        # Check convergence
        if abs(prev_loss - loss) < convergence_threshold:
            if verbose:
                print(f"\nConverged at iteration {i}")
            return OptimizationResult(
                final_params=params,
                final_loss=float(loss),
                history=history.to_dict(),
                converged=True,
                num_iterations=i + 1,
            )
        
        # Gradient descent step
        params = {k: v - learning_rate * gradient[k] for k, v in params.items()}
        
        prev_loss = loss
        
        if verbose and i % 10 == 0:
            iterator.set_postfix({'loss': f'{loss:.4f}'})
    
    final_loss = loss_fn(params)
    return OptimizationResult(
        final_params=params,
        final_loss=float(final_loss),
        history=history.to_dict(),
        converged=False,
        num_iterations=num_iterations,
    )


def optimize_adam(
    loss_fn: Callable,
    initial_params: dict,
    learning_rate: float = 0.01,
    num_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize using Adam optimizer (via optax).
    
    Adam typically converges faster than vanilla gradient descent.
    """
    params = initial_params.copy()
    history = OptimizationHistory()
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads
    
    prev_loss = float('inf')
    
    iterator = range(num_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="Optimizing (Adam)")
    
    for i in iterator:
        params, opt_state, loss, grads = step(params, opt_state)
        
        history.record(float(loss), params.copy(), grads)
        
        if abs(prev_loss - float(loss)) < convergence_threshold:
            if verbose:
                print(f"\nConverged at iteration {i}")
            return OptimizationResult(
                final_params=params,
                final_loss=float(loss),
                history=history.to_dict(),
                converged=True,
                num_iterations=i + 1,
            )
        
        prev_loss = float(loss)
        
        if verbose and i % 10 == 0:
            iterator.set_postfix({'loss': f'{float(loss):.4f}'})
    
    final_loss = loss_fn(params)
    return OptimizationResult(
        final_params=params,
        final_loss=float(final_loss),
        history=history.to_dict(),
        converged=False,
        num_iterations=num_iterations,
    )


def optimize_lbfgs(
    loss_fn: Callable,
    initial_params: dict,
    num_iterations: int = 50,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize using L-BFGS (via optax).
    
    L-BFGS is a quasi-Newton method that can be very efficient
    for smooth optimization problems.
    """
    params = initial_params.copy()
    history = OptimizationHistory()
    
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(params)
    
    value_and_grad_fn = value_and_grad(loss_fn)
    
    for i in range(num_iterations):
        loss, grads = value_and_grad_fn(params)
        
        history.record(float(loss), params.copy(), grads)
        
        updates, opt_state = optimizer.update(
            grads, opt_state, params,
            value=loss, grad=grads, value_fn=loss_fn
        )
        params = optax.apply_updates(params, updates)
        
        if verbose and i % 10 == 0:
            print(f"Iteration {i}: loss = {float(loss):.6f}")
    
    final_loss = loss_fn(params)
    return OptimizationResult(
        final_params=params,
        final_loss=float(final_loss),
        history=history.to_dict(),
        converged=False,
        num_iterations=num_iterations,
    )


# =============================================================================
# High-Level Optimization Functions
# =============================================================================

def find_optimal_wind(
    initial_state: PhysicsState,
    target_trajectory: jnp.ndarray,
    config: SimulationConfig,
    initial_guess: dict = None,
    optimizer: str = 'adam',
    num_iterations: int = 200,
    learning_rate: float = 0.05,
    verbose: bool = True,
) -> tuple[WindField, OptimizationResult]:
    """
    Find optimal wind field parameters to match a target trajectory.
    
    Args:
        initial_state: Starting physics state
        target_trajectory: Target trajectory to match
        config: Simulation configuration
        initial_guess: Initial parameter guess (uses defaults if None)
        optimizer: Optimizer to use ('gd', 'adam', 'lbfgs')
        num_iterations: Maximum iterations
        learning_rate: Learning rate for optimizer
        verbose: Whether to print progress
        
    Returns:
        Tuple of (optimized WindField, OptimizationResult)
    """
    if initial_guess is None:
        # Smart initial guess: center in middle of trajectory
        traj_center = jnp.mean(target_trajectory, axis=0)
        initial_guess = create_wind_params(
            center=tuple(traj_center),
            size=(3.0, 3.0),
            direction=(0.0, 1.0),
            strength=5.0,
        )
    
    loss_fn = make_trajectory_loss_fn(
        initial_state, target_trajectory, config, field_type='wind'
    )
    
    if optimizer == 'gd':
        result = optimize_gradient_descent(
            loss_fn, initial_guess, learning_rate, num_iterations, verbose=verbose
        )
    elif optimizer == 'adam':
        result = optimize_adam(
            loss_fn, initial_guess, learning_rate, num_iterations, verbose=verbose
        )
    elif optimizer == 'lbfgs':
        result = optimize_lbfgs(
            loss_fn, initial_guess, num_iterations, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    optimal_field = params_to_wind_field(result.final_params)
    return optimal_field, result


def find_optimal_vortex(
    initial_state: PhysicsState,
    target_trajectory: jnp.ndarray,
    config: SimulationConfig,
    initial_guess: dict = None,
    optimizer: str = 'adam',
    num_iterations: int = 200,
    learning_rate: float = 0.05,
    verbose: bool = True,
) -> tuple[VortexField, OptimizationResult]:
    """Find optimal vortex field parameters to match a target trajectory."""
    if initial_guess is None:
        traj_center = jnp.mean(target_trajectory, axis=0)
        initial_guess = create_vortex_params(
            center=tuple(traj_center),
            strength=5.0,
            radius=2.0,
        )
    
    loss_fn = make_trajectory_loss_fn(
        initial_state, target_trajectory, config, field_type='vortex'
    )
    
    if optimizer == 'adam':
        result = optimize_adam(
            loss_fn, initial_guess, learning_rate, num_iterations, verbose=verbose
        )
    else:
        result = optimize_gradient_descent(
            loss_fn, initial_guess, learning_rate, num_iterations, verbose=verbose
        )
    
    optimal_field = params_to_vortex_field(result.final_params)
    return optimal_field, result


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_optimization_progress(history: dict, ax=None):
    """Plot loss over optimization iterations."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['losses'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Optimization Progress')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    return ax


def plot_trajectory_comparison(
    target: jnp.ndarray,
    optimized: jnp.ndarray,
    initial: jnp.ndarray = None,
    ax=None,
):
    """Plot comparison of target, initial, and optimized trajectories."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(target[:, 0], target[:, 1], 'g--', linewidth=2, label='Target')
    ax.plot(optimized[:, 0], optimized[:, 1], 'b-', linewidth=2, label='Optimized')
    
    if initial is not None:
        ax.plot(initial[:, 0], initial[:, 1], 'r:', linewidth=2, label='Initial', alpha=0.7)
    
    ax.scatter([target[0, 0]], [target[0, 1]], c='green', s=100, zorder=5, label='Start')
    ax.scatter([target[-1, 0]], [target[-1, 1]], c='red', s=100, zorder=5, label='End')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax

