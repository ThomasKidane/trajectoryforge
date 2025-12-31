"""
Data Augmentation

Augmentation techniques for trajectory data to improve model generalization.
"""

from __future__ import annotations
import jax.numpy as jnp
from jax import Array, random


def augment_trajectory(
    trajectory: Array,
    key: Array,
    flip_x: bool = True,
    flip_y: bool = True,
    rotate: bool = True,
    scale: bool = True,
    noise: bool = True,
    noise_std: float = 0.02,
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> Array:
    """
    Apply random augmentations to a trajectory.
    
    Args:
        trajectory: Input trajectory, shape (N, 2)
        key: JAX random key
        flip_x: Whether to randomly flip horizontally
        flip_y: Whether to randomly flip vertically
        rotate: Whether to randomly rotate
        scale: Whether to randomly scale
        noise: Whether to add Gaussian noise
        noise_std: Standard deviation of noise
        scale_range: Range for random scaling
        
    Returns:
        Augmented trajectory, shape (N, 2)
    """
    keys = random.split(key, 6)
    
    # Center the trajectory
    centroid = jnp.mean(trajectory, axis=0)
    traj = trajectory - centroid
    
    # Random horizontal flip
    if flip_x:
        do_flip = random.bernoulli(keys[0], 0.5)
        traj = jnp.where(do_flip, traj.at[:, 0].set(-traj[:, 0]), traj)
    
    # Random vertical flip
    if flip_y:
        do_flip = random.bernoulli(keys[1], 0.5)
        traj = jnp.where(do_flip, traj.at[:, 1].set(-traj[:, 1]), traj)
    
    # Random rotation
    if rotate:
        angle = random.uniform(keys[2], (), minval=0, maxval=2 * jnp.pi)
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        rot_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        traj = jnp.dot(traj, rot_matrix.T)
    
    # Random scaling
    if scale:
        scale_factor = random.uniform(keys[3], (), minval=scale_range[0], maxval=scale_range[1])
        traj = traj * scale_factor
    
    # Add noise
    if noise:
        traj = traj + random.normal(keys[4], traj.shape) * noise_std
    
    # Restore centroid (with random offset)
    if scale:
        offset = random.uniform(keys[5], (2,), minval=-0.5, maxval=0.5)
        traj = traj + centroid + offset
    else:
        traj = traj + centroid
    
    return traj


def augment_batch(
    trajectories: Array,
    key: Array,
    **kwargs,
) -> Array:
    """
    Apply augmentations to a batch of trajectories.
    
    Args:
        trajectories: Batch of trajectories, shape (batch, N, 2)
        key: JAX random key
        **kwargs: Arguments for augment_trajectory
        
    Returns:
        Augmented batch, shape (batch, N, 2)
    """
    batch_size = trajectories.shape[0]
    keys = random.split(key, batch_size)
    
    augmented = []
    for i in range(batch_size):
        aug_traj = augment_trajectory(trajectories[i], keys[i], **kwargs)
        augmented.append(aug_traj)
    
    return jnp.stack(augmented)


def temporal_subsample(
    trajectory: Array,
    key: Array,
    min_points: int = 50,
    max_points: int = 200,
) -> Array:
    """
    Randomly subsample points from trajectory.
    
    Args:
        trajectory: Input trajectory, shape (N, 2)
        key: JAX random key
        min_points: Minimum number of points to keep
        max_points: Maximum number of points to keep
        
    Returns:
        Subsampled trajectory
    """
    n_points = trajectory.shape[0]
    
    # Random number of points to keep
    keys = random.split(key, 2)
    n_keep = random.randint(keys[0], (), min_points, min(max_points, n_points) + 1)
    
    # Uniform sampling of indices
    indices = jnp.linspace(0, n_points - 1, n_keep).astype(jnp.int32)
    
    return trajectory[indices]


def temporal_jitter(
    trajectory: Array,
    key: Array,
    jitter_std: float = 0.1,
) -> Array:
    """
    Add temporal jitter by slightly shifting point timings.
    
    Args:
        trajectory: Input trajectory, shape (N, 2)
        key: JAX random key
        jitter_std: Standard deviation of jitter (as fraction of spacing)
        
    Returns:
        Jittered trajectory
    """
    n_points = trajectory.shape[0]
    
    # Generate jitter offsets
    jitter = random.normal(key, (n_points,)) * jitter_std
    
    # Create new indices with jitter
    base_indices = jnp.arange(n_points, dtype=jnp.float32)
    jittered_indices = base_indices + jitter
    jittered_indices = jnp.clip(jittered_indices, 0, n_points - 1)
    
    # Interpolate to get new positions
    x_interp = jnp.interp(jittered_indices, base_indices, trajectory[:, 0])
    y_interp = jnp.interp(jittered_indices, base_indices, trajectory[:, 1])
    
    return jnp.stack([x_interp, y_interp], axis=1)

