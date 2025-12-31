"""
Data Generator

Generates random force field configurations and corresponding trajectories
for training the inverse model.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, random

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField, VortexField, PointForce, DragZone, AnyField
from physics.simulator import simulate_positions_only


class DataSample(NamedTuple):
    """A single training sample."""
    trajectory: Array           # (num_steps, 2)
    initial_position: Array     # (2,)
    initial_velocity: Array     # (2,)
    field_params: dict          # Serialized field parameters


def generate_random_wind_field(
    key: Array,
    position_range: tuple[float, float] = (-4.0, 4.0),
    size_range: tuple[float, float] = (0.5, 3.0),
    strength_range: tuple[float, float] = (1.0, 15.0),
) -> WindField:
    """Generate a random wind field."""
    keys = random.split(key, 5)
    
    center = random.uniform(keys[0], (2,), minval=position_range[0], maxval=position_range[1])
    size = random.uniform(keys[1], (2,), minval=size_range[0], maxval=size_range[1])
    
    # Random direction (normalized)
    direction = random.normal(keys[2], (2,))
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    
    strength = random.uniform(keys[3], (), minval=strength_range[0], maxval=strength_range[1])
    
    # Random sign for strength
    sign = random.choice(keys[4], jnp.array([-1.0, 1.0]))
    
    return WindField(
        center=center,
        size=size,
        direction=direction,
        strength=float(strength * sign),
        softness=0.1,
    )


def generate_random_vortex_field(
    key: Array,
    position_range: tuple[float, float] = (-4.0, 4.0),
    strength_range: tuple[float, float] = (1.0, 10.0),
    radius_range: tuple[float, float] = (0.5, 2.5),
) -> VortexField:
    """Generate a random vortex field."""
    keys = random.split(key, 4)
    
    center = random.uniform(keys[0], (2,), minval=position_range[0], maxval=position_range[1])
    strength = random.uniform(keys[1], (), minval=strength_range[0], maxval=strength_range[1])
    radius = random.uniform(keys[2], (), minval=radius_range[0], maxval=radius_range[1])
    
    # Random sign for strength (CCW vs CW)
    sign = random.choice(keys[3], jnp.array([-1.0, 1.0]))
    
    return VortexField(
        center=center,
        strength=float(strength * sign),
        radius=float(radius),
        falloff="gaussian",
        softness=0.1,
    )


def generate_random_point_force(
    key: Array,
    position_range: tuple[float, float] = (-4.0, 4.0),
    strength_range: tuple[float, float] = (1.0, 8.0),
) -> PointForce:
    """Generate a random point force (attractor or repeller)."""
    keys = random.split(key, 3)
    
    center = random.uniform(keys[0], (2,), minval=position_range[0], maxval=position_range[1])
    strength = random.uniform(keys[1], (), minval=strength_range[0], maxval=strength_range[1])
    
    # Random sign (attractor vs repeller)
    sign = random.choice(keys[2], jnp.array([-1.0, 1.0]))
    
    return PointForce(
        center=center,
        strength=float(strength * sign),
        falloff_power=2.0,
        max_distance=10.0,
        softness=0.2,
    )


def generate_random_fields(
    key: Array,
    num_wind: int = 1,
    num_vortex: int = 1,
    num_point: int = 0,
    position_range: tuple[float, float] = (-4.0, 4.0),
) -> list[AnyField]:
    """
    Generate a random collection of force fields.
    
    Args:
        key: JAX random key
        num_wind: Number of wind fields
        num_vortex: Number of vortex fields
        num_point: Number of point forces
        position_range: Range for field positions
        
    Returns:
        List of force field objects
    """
    fields = []
    keys = random.split(key, num_wind + num_vortex + num_point)
    key_idx = 0
    
    for _ in range(num_wind):
        fields.append(generate_random_wind_field(keys[key_idx], position_range=position_range))
        key_idx += 1
    
    for _ in range(num_vortex):
        fields.append(generate_random_vortex_field(keys[key_idx], position_range=position_range))
        key_idx += 1
    
    for _ in range(num_point):
        fields.append(generate_random_point_force(keys[key_idx], position_range=position_range))
        key_idx += 1
    
    return fields


def generate_trajectory_from_fields(
    fields: list[AnyField],
    initial_position: Array | None = None,
    initial_velocity: Array | None = None,
    config: SimulationConfig | None = None,
    key: Array | None = None,
) -> Array:
    """
    Generate a trajectory from a set of force fields.
    
    Args:
        fields: List of force fields
        initial_position: Starting position (random if None)
        initial_velocity: Starting velocity (random if None)
        config: Simulation configuration
        key: Random key for initial conditions
        
    Returns:
        Trajectory array, shape (num_steps, 2)
    """
    if config is None:
        config = SimulationConfig(
            dt=0.01,
            num_steps=300,
            bounds=((-5.0, 5.0), (-5.0, 5.0)),
        )
    
    if key is None:
        key = random.PRNGKey(0)
    
    keys = random.split(key, 2)
    
    if initial_position is None:
        initial_position = random.uniform(keys[0], (2,), minval=-3.0, maxval=-1.0)
    
    if initial_velocity is None:
        # Random velocity with magnitude between 1 and 2
        direction = random.normal(keys[1], (2,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-8)
        speed = random.uniform(keys[1], (), minval=1.0, maxval=2.0)
        initial_velocity = direction * speed
    
    state = PhysicsState(
        position=initial_position,
        velocity=initial_velocity,
        time=0.0,
    )
    
    trajectory = simulate_positions_only(state, fields, config)
    
    return trajectory


def generate_dataset_sample(
    key: Array,
    num_wind: int = 1,
    num_vortex: int = 1,
    num_point: int = 0,
    config: SimulationConfig | None = None,
) -> DataSample:
    """
    Generate a single dataset sample.
    
    Args:
        key: JAX random key
        num_wind: Number of wind fields
        num_vortex: Number of vortex fields
        num_point: Number of point forces
        config: Simulation configuration
        
    Returns:
        DataSample with trajectory and field parameters
    """
    keys = random.split(key, 3)
    
    # Generate random fields
    fields = generate_random_fields(
        keys[0],
        num_wind=num_wind,
        num_vortex=num_vortex,
        num_point=num_point,
    )
    
    # Generate random initial conditions
    initial_position = random.uniform(keys[1], (2,), minval=-3.0, maxval=-1.0)
    
    direction = random.normal(keys[2], (2,))
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    speed = random.uniform(keys[2], (), minval=1.0, maxval=2.0)
    initial_velocity = direction * speed
    
    # Generate trajectory
    trajectory = generate_trajectory_from_fields(
        fields,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        config=config,
    )
    
    # Serialize field parameters
    from physics.fields import fields_to_params
    field_params = {
        'fields': fields_to_params(fields),
        'initial_position': initial_position.tolist(),
        'initial_velocity': initial_velocity.tolist(),
    }
    
    return DataSample(
        trajectory=trajectory,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        field_params=field_params,
    )


def generate_dataset(
    num_samples: int,
    seed: int = 42,
    num_wind: int = 1,
    num_vortex: int = 1,
    num_point: int = 0,
    config: SimulationConfig | None = None,
) -> list[DataSample]:
    """
    Generate a dataset of trajectory samples.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed
        num_wind: Number of wind fields per sample
        num_vortex: Number of vortex fields per sample
        num_point: Number of point forces per sample
        config: Simulation configuration
        
    Returns:
        List of DataSample objects
    """
    key = random.PRNGKey(seed)
    keys = random.split(key, num_samples)
    
    samples = []
    for i in range(num_samples):
        sample = generate_dataset_sample(
            keys[i],
            num_wind=num_wind,
            num_vortex=num_vortex,
            num_point=num_point,
            config=config,
        )
        samples.append(sample)
    
    return samples

