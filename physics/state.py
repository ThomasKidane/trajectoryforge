"""
Physics State Definitions

This module defines the core data structures for the physics simulation,
including the physics state and simulation configuration.

All structures are designed to work seamlessly with JAX for automatic differentiation.
"""

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
from jax import Array


class PhysicsState(NamedTuple):
    """
    Represents the instantaneous state of the physics simulation.
    
    This is a NamedTuple for JAX compatibility - it can be used as a pytree
    and supports automatic differentiation through all operations.
    
    Attributes:
        position: Ball position as (x, y) coordinates
        velocity: Ball velocity as (vx, vy) components
        time: Current simulation time in seconds
    
    Example:
        >>> state = PhysicsState(
        ...     position=jnp.array([0.0, 0.0]),
        ...     velocity=jnp.array([1.0, 0.0]),
        ...     time=0.0
        ... )
        >>> state.position
        Array([0., 0.], dtype=float32)
    """
    position: Array  # Shape: (2,) - [x, y]
    velocity: Array  # Shape: (2,) - [vx, vy]
    time: float      # Simulation time in seconds
    
    @staticmethod
    def create(
        position: tuple[float, float] = (0.0, 0.0),
        velocity: tuple[float, float] = (0.0, 0.0),
        time: float = 0.0
    ) -> PhysicsState:
        """
        Factory method to create a PhysicsState from Python tuples.
        
        Args:
            position: Initial (x, y) position
            velocity: Initial (vx, vy) velocity
            time: Initial simulation time
            
        Returns:
            A new PhysicsState instance
        """
        return PhysicsState(
            position=jnp.array(position, dtype=jnp.float32),
            velocity=jnp.array(velocity, dtype=jnp.float32),
            time=float(time)
        )
    
    def kinetic_energy(self, mass: float = 1.0) -> float:
        """Compute kinetic energy: KE = 0.5 * m * v^2"""
        return 0.5 * mass * jnp.sum(self.velocity ** 2)
    
    def speed(self) -> float:
        """Compute the magnitude of velocity."""
        return jnp.linalg.norm(self.velocity)


class SimulationConfig(NamedTuple):
    """
    Configuration parameters for the physics simulation.
    
    Attributes:
        dt: Time step for integration (seconds)
        num_steps: Number of simulation steps
        ball_mass: Mass of the ball (kg)
        ball_radius: Radius of the ball for collision detection (m)
        bounds: Simulation bounds as ((x_min, x_max), (y_min, y_max))
        boundary_softness: Softness of boundary repulsion (higher = softer)
        boundary_strength: Strength of boundary repulsion force
        global_gravity: Global gravity vector (usually (0, -9.81) or (0, 0))
        drag_coefficient: Global air resistance coefficient
    """
    dt: float = 0.01                    # 10ms timestep (100Hz simulation)
    num_steps: int = 500                # 5 seconds of simulation
    ball_mass: float = 1.0              # 1 kg
    ball_radius: float = 0.05           # 5 cm radius
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-5.0, 5.0), (-5.0, 5.0))
    boundary_softness: float = 0.1      # Soft boundary transition width
    boundary_strength: float = 100.0    # Boundary repulsion strength
    global_gravity: tuple[float, float] = (0.0, 0.0)  # No gravity by default
    drag_coefficient: float = 0.0       # No global drag by default
    
    @property
    def total_time(self) -> float:
        """Total simulation duration in seconds."""
        return self.dt * self.num_steps
    
    @property 
    def bounds_array(self) -> Array:
        """Return bounds as JAX array for efficient computation."""
        return jnp.array(self.bounds, dtype=jnp.float32)
    
    @property
    def gravity_array(self) -> Array:
        """Return gravity as JAX array."""
        return jnp.array(self.global_gravity, dtype=jnp.float32)


class TrajectoryData(NamedTuple):
    """
    Stores a complete trajectory from a simulation.
    
    Attributes:
        positions: Array of positions, shape (num_steps, 2)
        velocities: Array of velocities, shape (num_steps, 2)
        times: Array of timestamps, shape (num_steps,)
        forces: Optional array of forces at each step, shape (num_steps, 2)
    """
    positions: Array    # Shape: (num_steps, 2)
    velocities: Array   # Shape: (num_steps, 2)
    times: Array        # Shape: (num_steps,)
    forces: Array | None = None  # Shape: (num_steps, 2), optional
    
    @property
    def num_points(self) -> int:
        """Number of points in the trajectory."""
        return self.positions.shape[0]
    
    @property
    def duration(self) -> float:
        """Total duration of the trajectory."""
        return float(self.times[-1] - self.times[0])
    
    def path_length(self) -> float:
        """Compute the total arc length of the trajectory."""
        diffs = jnp.diff(self.positions, axis=0)
        segment_lengths = jnp.linalg.norm(diffs, axis=1)
        return jnp.sum(segment_lengths)


# Type aliases for clarity
Position = Array  # Shape: (2,)
Velocity = Array  # Shape: (2,)
Force = Array     # Shape: (2,)
Trajectory = Array  # Shape: (N, 2)


def create_initial_state(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 1.0,
    vy: float = 0.0
) -> PhysicsState:
    """
    Convenience function to create an initial physics state.
    
    Args:
        x: Initial x position
        y: Initial y position
        vx: Initial x velocity
        vy: Initial y velocity
        
    Returns:
        A new PhysicsState
    """
    return PhysicsState.create(
        position=(x, y),
        velocity=(vx, vy),
        time=0.0
    )


def default_config() -> SimulationConfig:
    """Return the default simulation configuration."""
    return SimulationConfig()


def game_config() -> SimulationConfig:
    """
    Return a simulation configuration suitable for the game.
    
    This uses a larger play area and more steps for longer trajectories.
    """
    return SimulationConfig(
        dt=0.016,           # ~60Hz for smooth animation
        num_steps=600,      # 10 seconds of simulation
        ball_mass=1.0,
        ball_radius=0.1,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        boundary_softness=0.2,
        boundary_strength=50.0,
        global_gravity=(0.0, 0.0),
        drag_coefficient=0.01
    )

