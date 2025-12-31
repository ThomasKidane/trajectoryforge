"""
Data Augmentation Module

Provides augmentation techniques for trajectory data to improve
model generalization and training efficiency.
"""

from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np


# =============================================================================
# Geometric Augmentations
# =============================================================================

def augment_translate(
    key: jax.Array,
    trajectory: jnp.ndarray,
    max_offset: float = 1.0,
) -> jnp.ndarray:
    """
    Randomly translate a trajectory.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        max_offset: Maximum translation in each direction
        
    Returns:
        Translated trajectory
    """
    offset = random.uniform(key, (2,), minval=-max_offset, maxval=max_offset)
    return trajectory + offset


def augment_scale(
    key: jax.Array,
    trajectory: jnp.ndarray,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    uniform: bool = True,
) -> jnp.ndarray:
    """
    Randomly scale a trajectory around its centroid.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        uniform: If True, use same scale for x and y
        
    Returns:
        Scaled trajectory
    """
    centroid = jnp.mean(trajectory, axis=0)
    
    if uniform:
        scale = random.uniform(key, (), minval=min_scale, maxval=max_scale)
        scale = jnp.array([scale, scale])
    else:
        scale = random.uniform(key, (2,), minval=min_scale, maxval=max_scale)
    
    centered = trajectory - centroid
    scaled = centered * scale
    return scaled + centroid


def augment_rotate(
    key: jax.Array,
    trajectory: jnp.ndarray,
    max_angle: float = jnp.pi / 4,  # 45 degrees
) -> jnp.ndarray:
    """
    Randomly rotate a trajectory around its centroid.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        max_angle: Maximum rotation angle in radians
        
    Returns:
        Rotated trajectory
    """
    centroid = jnp.mean(trajectory, axis=0)
    angle = random.uniform(key, (), minval=-max_angle, maxval=max_angle)
    
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    rot_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    centered = trajectory - centroid
    rotated = jnp.dot(centered, rot_matrix.T)
    return rotated + centroid


def augment_flip(
    key: jax.Array,
    trajectory: jnp.ndarray,
    p_flip_x: float = 0.5,
    p_flip_y: float = 0.5,
) -> jnp.ndarray:
    """
    Randomly flip a trajectory along x and/or y axes.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        p_flip_x: Probability of flipping along x axis
        p_flip_y: Probability of flipping along y axis
        
    Returns:
        Flipped trajectory
    """
    key1, key2 = random.split(key)
    centroid = jnp.mean(trajectory, axis=0)
    centered = trajectory - centroid
    
    flip_x = random.bernoulli(key1, p_flip_x)
    flip_y = random.bernoulli(key2, p_flip_y)
    
    scale_x = jnp.where(flip_x, -1.0, 1.0)
    scale_y = jnp.where(flip_y, -1.0, 1.0)
    
    flipped = centered * jnp.array([scale_x, scale_y])
    return flipped + centroid


def augment_shear(
    key: jax.Array,
    trajectory: jnp.ndarray,
    max_shear: float = 0.2,
) -> jnp.ndarray:
    """
    Apply random shear transformation to a trajectory.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        max_shear: Maximum shear factor
        
    Returns:
        Sheared trajectory
    """
    key1, key2 = random.split(key)
    centroid = jnp.mean(trajectory, axis=0)
    centered = trajectory - centroid
    
    shear_x = random.uniform(key1, (), minval=-max_shear, maxval=max_shear)
    shear_y = random.uniform(key2, (), minval=-max_shear, maxval=max_shear)
    
    shear_matrix = jnp.array([[1.0, shear_x], [shear_y, 1.0]])
    sheared = jnp.dot(centered, shear_matrix.T)
    
    return sheared + centroid


# =============================================================================
# Temporal Augmentations
# =============================================================================

def augment_reverse(
    key: jax.Array,
    trajectory: jnp.ndarray,
    p_reverse: float = 0.5,
) -> jnp.ndarray:
    """
    Randomly reverse trajectory direction.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        p_reverse: Probability of reversing
        
    Returns:
        Possibly reversed trajectory
    """
    should_reverse = random.bernoulli(key, p_reverse)
    return jnp.where(should_reverse, trajectory[::-1], trajectory)


def augment_subsample(
    key: jax.Array,
    trajectory: jnp.ndarray,
    min_keep: float = 0.7,
    max_keep: float = 1.0,
) -> jnp.ndarray:
    """
    Randomly subsample trajectory points (keep a fraction).
    
    Note: This changes trajectory length, so may need resampling after.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        min_keep: Minimum fraction of points to keep
        max_keep: Maximum fraction of points to keep
        
    Returns:
        Subsampled trajectory
    """
    n_points = trajectory.shape[0]
    key1, key2 = random.split(key)
    
    keep_frac = random.uniform(key1, (), minval=min_keep, maxval=max_keep)
    n_keep = jnp.maximum(2, jnp.int32(n_points * keep_frac))
    
    # Generate sorted random indices
    indices = jnp.sort(random.choice(key2, n_points, shape=(n_keep,), replace=False))
    
    return trajectory[indices]


def augment_time_warp(
    key: jax.Array,
    trajectory: jnp.ndarray,
    warp_strength: float = 0.2,
) -> jnp.ndarray:
    """
    Apply random time warping to trajectory.
    
    This stretches/compresses different parts of the trajectory
    while keeping the same start and end points.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        warp_strength: Strength of warping
        
    Returns:
        Time-warped trajectory
    """
    n_points = trajectory.shape[0]
    
    # Generate warped time indices
    t_original = jnp.linspace(0, 1, n_points)
    
    # Add smooth random perturbations
    key1, key2 = random.split(key)
    n_control = 5
    control_points = random.normal(key1, (n_control,)) * warp_strength
    control_points = control_points.at[0].set(0)  # Fix start
    control_points = control_points.at[-1].set(0)  # Fix end
    
    # Interpolate perturbations
    t_control = jnp.linspace(0, 1, n_control)
    perturbations = jnp.interp(t_original, t_control, control_points)
    
    # Apply warping (monotonic)
    t_warped = t_original + perturbations
    t_warped = jnp.clip(t_warped, 0, 1)
    t_warped = jnp.sort(t_warped)  # Ensure monotonicity
    
    # Resample trajectory at warped times
    x_warped = jnp.interp(t_original, t_warped, trajectory[:, 0])
    y_warped = jnp.interp(t_original, t_warped, trajectory[:, 1])
    
    return jnp.stack([x_warped, y_warped], axis=1)


# =============================================================================
# Noise Augmentations
# =============================================================================

def augment_gaussian_noise(
    key: jax.Array,
    trajectory: jnp.ndarray,
    noise_std: float = 0.05,
) -> jnp.ndarray:
    """
    Add Gaussian noise to trajectory points.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        noise_std: Standard deviation of noise
        
    Returns:
        Noisy trajectory
    """
    noise = random.normal(key, trajectory.shape) * noise_std
    return trajectory + noise


def augment_smooth_noise(
    key: jax.Array,
    trajectory: jnp.ndarray,
    noise_std: float = 0.1,
    smoothness: int = 5,
) -> jnp.ndarray:
    """
    Add smooth (correlated) noise to trajectory.
    
    This creates more realistic perturbations than i.i.d. Gaussian noise.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        noise_std: Standard deviation of noise
        smoothness: Kernel size for smoothing (larger = smoother)
        
    Returns:
        Noisy trajectory
    """
    n_points = trajectory.shape[0]
    
    # Generate noise at control points
    key1, key2 = random.split(key)
    n_control = max(3, n_points // smoothness)
    control_noise = random.normal(key1, (n_control, 2)) * noise_std
    
    # Interpolate to full resolution
    t_control = jnp.linspace(0, 1, n_control)
    t_full = jnp.linspace(0, 1, n_points)
    
    noise_x = jnp.interp(t_full, t_control, control_noise[:, 0])
    noise_y = jnp.interp(t_full, t_control, control_noise[:, 1])
    noise = jnp.stack([noise_x, noise_y], axis=1)
    
    return trajectory + noise


def augment_dropout(
    key: jax.Array,
    trajectory: jnp.ndarray,
    drop_rate: float = 0.1,
) -> jnp.ndarray:
    """
    Randomly drop trajectory points and interpolate.
    
    Simulates missing data or sensor dropout.
    
    Args:
        key: JAX random key
        trajectory: Input trajectory, shape (N, 2)
        drop_rate: Fraction of points to drop
        
    Returns:
        Trajectory with dropped points interpolated
    """
    n_points = trajectory.shape[0]
    key1, key2 = random.split(key)
    
    # Generate mask (keep first and last points)
    mask = random.bernoulli(key1, 1 - drop_rate, (n_points,))
    mask = mask.at[0].set(True)
    mask = mask.at[-1].set(True)
    
    # Get indices of kept points
    kept_indices = jnp.where(mask, jnp.arange(n_points), -1)
    kept_indices = kept_indices[kept_indices >= 0]
    
    # Interpolate to fill gaps
    t_kept = kept_indices / (n_points - 1)
    t_full = jnp.linspace(0, 1, n_points)
    
    x_interp = jnp.interp(t_full, t_kept, trajectory[kept_indices, 0])
    y_interp = jnp.interp(t_full, t_kept, trajectory[kept_indices, 1])
    
    return jnp.stack([x_interp, y_interp], axis=1)


# =============================================================================
# Composite Augmentation Pipeline
# =============================================================================

class AugmentationPipeline:
    """
    Composable augmentation pipeline for trajectory data.
    
    Applies a sequence of augmentations with configurable probabilities.
    """
    
    def __init__(
        self,
        augmentations: list[tuple[Callable, dict, float]] = None,
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of (augment_fn, kwargs, probability) tuples.
                          If None, uses default augmentations.
        """
        if augmentations is None:
            self.augmentations = self._default_augmentations()
        else:
            self.augmentations = augmentations
    
    def _default_augmentations(self) -> list[tuple[Callable, dict, float]]:
        """Get default augmentation configuration."""
        return [
            (augment_translate, {'max_offset': 0.5}, 0.5),
            (augment_scale, {'min_scale': 0.9, 'max_scale': 1.1}, 0.5),
            (augment_rotate, {'max_angle': jnp.pi / 6}, 0.5),
            (augment_flip, {'p_flip_x': 0.3, 'p_flip_y': 0.3}, 0.5),
            (augment_gaussian_noise, {'noise_std': 0.02}, 0.5),
        ]
    
    def __call__(
        self,
        key: jax.Array,
        trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply augmentation pipeline to a trajectory.
        
        Args:
            key: JAX random key
            trajectory: Input trajectory, shape (N, 2)
            
        Returns:
            Augmented trajectory
        """
        for aug_fn, kwargs, prob in self.augmentations:
            key, subkey1, subkey2 = random.split(key, 3)
            
            # Decide whether to apply this augmentation
            apply_aug = random.bernoulli(subkey1, prob)
            
            if apply_aug:
                trajectory = aug_fn(subkey2, trajectory, **kwargs)
        
        return trajectory
    
    def apply_batch(
        self,
        key: jax.Array,
        trajectories: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply augmentation to a batch of trajectories.
        
        Args:
            key: JAX random key
            trajectories: Batch of trajectories, shape (B, N, 2)
            
        Returns:
            Augmented trajectories, shape (B, N, 2)
        """
        batch_size = trajectories.shape[0]
        keys = random.split(key, batch_size)
        
        # Apply augmentation to each trajectory
        augmented = []
        for i in range(batch_size):
            aug_traj = self(keys[i], trajectories[i])
            augmented.append(aug_traj)
        
        return jnp.stack(augmented)


def create_strong_augmentation() -> AugmentationPipeline:
    """Create a strong augmentation pipeline for aggressive data augmentation."""
    return AugmentationPipeline([
        (augment_translate, {'max_offset': 1.0}, 0.7),
        (augment_scale, {'min_scale': 0.7, 'max_scale': 1.3}, 0.7),
        (augment_rotate, {'max_angle': jnp.pi / 3}, 0.7),
        (augment_flip, {'p_flip_x': 0.5, 'p_flip_y': 0.5}, 0.5),
        (augment_shear, {'max_shear': 0.3}, 0.3),
        (augment_time_warp, {'warp_strength': 0.3}, 0.5),
        (augment_gaussian_noise, {'noise_std': 0.05}, 0.7),
        (augment_smooth_noise, {'noise_std': 0.1}, 0.3),
    ])


def create_light_augmentation() -> AugmentationPipeline:
    """Create a light augmentation pipeline for subtle variations."""
    return AugmentationPipeline([
        (augment_translate, {'max_offset': 0.2}, 0.3),
        (augment_scale, {'min_scale': 0.95, 'max_scale': 1.05}, 0.3),
        (augment_rotate, {'max_angle': jnp.pi / 12}, 0.3),
        (augment_gaussian_noise, {'noise_std': 0.01}, 0.3),
    ])


# =============================================================================
# Augmentation Utilities
# =============================================================================

def augment_sample(
    key: jax.Array,
    sample: dict,
    pipeline: AugmentationPipeline,
    augment_initial_conditions: bool = True,
) -> dict:
    """
    Augment a complete training sample.
    
    Args:
        key: JAX random key
        sample: Training sample dict with 'trajectory', 'initial_position', etc.
        pipeline: Augmentation pipeline to apply
        augment_initial_conditions: Whether to update initial pos/vel to match
        
    Returns:
        Augmented sample
    """
    key1, key2 = random.split(key)
    
    # Augment trajectory
    aug_trajectory = pipeline(key1, sample['trajectory'])
    
    # Create new sample
    aug_sample = {**sample, 'trajectory': aug_trajectory}
    
    # Update initial conditions if requested
    if augment_initial_conditions:
        # New initial position is first point of augmented trajectory
        aug_sample['initial_position'] = aug_trajectory[0]
        
        # Estimate new initial velocity from first two points
        if aug_trajectory.shape[0] > 1:
            dt = sample.get('dt', 0.01)
            aug_sample['initial_velocity'] = (aug_trajectory[1] - aug_trajectory[0]) / dt
    
    return aug_sample


def augment_dataset(
    key: jax.Array,
    dataset: list[dict],
    pipeline: AugmentationPipeline,
    num_augmentations: int = 1,
) -> list[dict]:
    """
    Augment an entire dataset.
    
    Args:
        key: JAX random key
        dataset: List of training samples
        pipeline: Augmentation pipeline
        num_augmentations: Number of augmented versions per sample
        
    Returns:
        Augmented dataset (original + augmented samples)
    """
    augmented = list(dataset)  # Keep originals
    
    for sample in dataset:
        keys = random.split(key, num_augmentations + 1)
        key = keys[0]
        
        for i in range(num_augmentations):
            aug_sample = augment_sample(keys[i + 1], sample, pipeline)
            augmented.append(aug_sample)
    
    return augmented

