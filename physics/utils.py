"""
Trajectory Utilities

This module provides utility functions for working with trajectories,
including interpolation, resampling, path generation, and analysis.
"""

from __future__ import annotations
from typing import Callable
import jax.numpy as jnp
from jax import Array
import numpy as np


# =============================================================================
# Trajectory Interpolation
# =============================================================================

def interpolate_trajectory(
    trajectory: Array,
    t: float | Array
) -> Array:
    """
    Interpolate a trajectory at parameter value(s) t in [0, 1].
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        t: Parameter value(s) in [0, 1], scalar or array
        
    Returns:
        Interpolated point(s), shape (2,) or (M, 2)
    """
    n_points = trajectory.shape[0]
    t_values = jnp.linspace(0, 1, n_points)
    
    # Handle scalar t
    t = jnp.atleast_1d(t)
    
    x_interp = jnp.interp(t, t_values, trajectory[:, 0])
    y_interp = jnp.interp(t, t_values, trajectory[:, 1])
    
    result = jnp.stack([x_interp, y_interp], axis=-1)
    
    # Return scalar if input was scalar
    if result.shape[0] == 1:
        return result[0]
    return result


def resample_trajectory(
    trajectory: Array,
    num_points: int
) -> Array:
    """
    Resample a trajectory to a specific number of points.
    
    Uses uniform parameter spacing for consistent point distribution.
    
    Args:
        trajectory: Input trajectory, shape (N, 2)
        num_points: Desired number of output points
        
    Returns:
        Resampled trajectory, shape (num_points, 2)
    """
    t_values = jnp.linspace(0, 1, num_points)
    return interpolate_trajectory(trajectory, t_values)


def resample_by_arc_length(
    trajectory: Array,
    num_points: int
) -> Array:
    """
    Resample trajectory with uniform arc length spacing.
    
    This produces more evenly spaced points along the actual path,
    regardless of the original parameterization.
    
    Args:
        trajectory: Input trajectory, shape (N, 2)
        num_points: Desired number of output points
        
    Returns:
        Resampled trajectory, shape (num_points, 2)
    """
    # Compute cumulative arc length
    diffs = jnp.diff(trajectory, axis=0)
    segment_lengths = jnp.linalg.norm(diffs, axis=1)
    cumulative_length = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(segment_lengths)
    ])
    total_length = cumulative_length[-1]
    
    # Target arc lengths
    target_lengths = jnp.linspace(0, total_length, num_points)
    
    # Interpolate to find points at target arc lengths
    # Normalize cumulative length to [0, 1] for interpolation
    t_original = cumulative_length / (total_length + 1e-8)
    t_target = target_lengths / (total_length + 1e-8)
    
    x_resampled = jnp.interp(t_target, t_original, trajectory[:, 0])
    y_resampled = jnp.interp(t_target, t_original, trajectory[:, 1])
    
    return jnp.stack([x_resampled, y_resampled], axis=1)


# =============================================================================
# Trajectory Analysis
# =============================================================================

def compute_trajectory_length(trajectory: Array) -> float:
    """
    Compute the total arc length of a trajectory.
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        
    Returns:
        Total arc length
    """
    diffs = jnp.diff(trajectory, axis=0)
    segment_lengths = jnp.linalg.norm(diffs, axis=1)
    return jnp.sum(segment_lengths)


def compute_curvature(trajectory: Array) -> Array:
    """
    Compute curvature at each point of a trajectory.
    
    Uses finite differences to estimate curvature.
    Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        
    Returns:
        Curvature values, shape (N-2,)
    """
    # First derivatives (velocity)
    dx = jnp.diff(trajectory[:, 0])
    dy = jnp.diff(trajectory[:, 1])
    
    # Second derivatives (acceleration)
    ddx = jnp.diff(dx)
    ddy = jnp.diff(dy)
    
    # Average first derivatives at interior points
    dx_mid = (dx[:-1] + dx[1:]) / 2
    dy_mid = (dy[:-1] + dy[1:]) / 2
    
    # Curvature formula
    numerator = jnp.abs(dx_mid * ddy - dy_mid * ddx)
    denominator = (dx_mid**2 + dy_mid**2) ** 1.5 + 1e-8
    
    return numerator / denominator


def compute_velocity_profile(trajectory: Array, dt: float) -> Array:
    """
    Compute velocity magnitude at each point.
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        dt: Time step between points
        
    Returns:
        Velocity magnitudes, shape (N-1,)
    """
    diffs = jnp.diff(trajectory, axis=0)
    return jnp.linalg.norm(diffs, axis=1) / dt


def compute_acceleration_profile(trajectory: Array, dt: float) -> Array:
    """
    Compute acceleration magnitude at each point.
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        dt: Time step between points
        
    Returns:
        Acceleration magnitudes, shape (N-2,)
    """
    velocities = jnp.diff(trajectory, axis=0) / dt
    accelerations = jnp.diff(velocities, axis=0) / dt
    return jnp.linalg.norm(accelerations, axis=1)


def trajectory_bounding_box(trajectory: Array) -> tuple[Array, Array]:
    """
    Compute the axis-aligned bounding box of a trajectory.
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        
    Returns:
        Tuple of (min_corner, max_corner), each shape (2,)
    """
    min_corner = jnp.min(trajectory, axis=0)
    max_corner = jnp.max(trajectory, axis=0)
    return min_corner, max_corner


def trajectory_centroid(trajectory: Array) -> Array:
    """
    Compute the centroid (center of mass) of a trajectory.
    
    Args:
        trajectory: Trajectory points, shape (N, 2)
        
    Returns:
        Centroid position, shape (2,)
    """
    return jnp.mean(trajectory, axis=0)


# =============================================================================
# Trajectory Generation
# =============================================================================

def generate_line_trajectory(
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int = 100
) -> Array:
    """
    Generate a straight line trajectory.
    
    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        num_points: Number of points
        
    Returns:
        Trajectory array, shape (num_points, 2)
    """
    start = jnp.array(start)
    end = jnp.array(end)
    t = jnp.linspace(0, 1, num_points)[:, None]
    return start + t * (end - start)


def generate_arc_trajectory(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    num_points: int = 100
) -> Array:
    """
    Generate a circular arc trajectory.
    
    Args:
        center: Arc center (x, y)
        radius: Arc radius
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians
        num_points: Number of points
        
    Returns:
        Trajectory array, shape (num_points, 2)
    """
    center = jnp.array(center)
    angles = jnp.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * jnp.cos(angles)
    y = center[1] + radius * jnp.sin(angles)
    return jnp.stack([x, y], axis=1)


def generate_spiral_trajectory(
    center: tuple[float, float],
    start_radius: float,
    end_radius: float,
    num_turns: float,
    num_points: int = 100
) -> Array:
    """
    Generate a spiral trajectory.
    
    Args:
        center: Spiral center (x, y)
        start_radius: Starting radius
        end_radius: Ending radius
        num_turns: Number of full rotations
        num_points: Number of points
        
    Returns:
        Trajectory array, shape (num_points, 2)
    """
    center = jnp.array(center)
    t = jnp.linspace(0, 1, num_points)
    angles = t * num_turns * 2 * jnp.pi
    radii = start_radius + t * (end_radius - start_radius)
    
    x = center[0] + radii * jnp.cos(angles)
    y = center[1] + radii * jnp.sin(angles)
    return jnp.stack([x, y], axis=1)


def generate_sine_wave_trajectory(
    start: tuple[float, float],
    end: tuple[float, float],
    amplitude: float,
    frequency: float,
    num_points: int = 100
) -> Array:
    """
    Generate a sinusoidal wave trajectory.
    
    The wave oscillates perpendicular to the line from start to end.
    
    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        amplitude: Wave amplitude
        frequency: Number of complete oscillations
        num_points: Number of points
        
    Returns:
        Trajectory array, shape (num_points, 2)
    """
    start = jnp.array(start)
    end = jnp.array(end)
    
    # Direction along the wave
    direction = end - start
    length = jnp.linalg.norm(direction)
    unit_dir = direction / (length + 1e-8)
    
    # Perpendicular direction
    perp_dir = jnp.array([-unit_dir[1], unit_dir[0]])
    
    # Generate points
    t = jnp.linspace(0, 1, num_points)
    base_points = start + t[:, None] * direction
    offsets = amplitude * jnp.sin(2 * jnp.pi * frequency * t)
    
    return base_points + offsets[:, None] * perp_dir


def generate_bezier_trajectory(
    control_points: list[tuple[float, float]],
    num_points: int = 100
) -> Array:
    """
    Generate a Bezier curve trajectory.
    
    Args:
        control_points: List of control points [(x1, y1), (x2, y2), ...]
        num_points: Number of output points
        
    Returns:
        Trajectory array, shape (num_points, 2)
    """
    points = jnp.array(control_points)
    n = len(control_points) - 1
    t = jnp.linspace(0, 1, num_points)
    
    # De Casteljau's algorithm
    def bezier_point(t_val):
        pts = points.copy()
        for r in range(1, n + 1):
            pts = pts[:n + 1 - r]
            pts = (1 - t_val) * pts[:-1] + t_val * pts[1:]
        return pts[0]
    
    # Vectorized computation using binomial coefficients
    # B(t) = sum_{i=0}^{n} C(n,i) * (1-t)^(n-i) * t^i * P_i
    from scipy.special import comb
    
    result = jnp.zeros((num_points, 2))
    for i in range(n + 1):
        coeff = comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)
        result = result + coeff[:, None] * points[i]
    
    return result


def generate_figure_eight_trajectory(
    center: tuple[float, float],
    width: float,
    height: float,
    num_points: int = 100
) -> Array:
    """
    Generate a figure-eight (lemniscate) trajectory.
    
    Args:
        center: Center of the figure eight
        width: Horizontal extent
        height: Vertical extent
        num_points: Number of points
        
    Returns:
        Trajectory array, shape (num_points, 2)
    """
    center = jnp.array(center)
    t = jnp.linspace(0, 2 * jnp.pi, num_points)
    
    # Parametric lemniscate
    x = center[0] + width * jnp.sin(t)
    y = center[1] + height * jnp.sin(t) * jnp.cos(t)
    
    return jnp.stack([x, y], axis=1)


def generate_random_smooth_trajectory(
    start: tuple[float, float],
    end: tuple[float, float],
    num_control_points: int = 5,
    smoothness: float = 0.5,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    seed: int | None = None
) -> Array:
    """
    Generate a random smooth trajectory using Bezier curves.
    
    Args:
        start: Starting point
        end: Ending point
        num_control_points: Number of intermediate control points
        smoothness: Controls how much control points deviate (0-1)
        bounds: Optional ((x_min, x_max), (y_min, y_max)) to constrain points
        seed: Random seed for reproducibility
        
    Returns:
        Trajectory array, shape (100, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    start = np.array(start)
    end = np.array(end)
    
    # Generate intermediate control points
    t_values = np.linspace(0, 1, num_control_points + 2)[1:-1]
    base_points = start + t_values[:, None] * (end - start)
    
    # Add random offsets
    direction = end - start
    length = np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]]) / (length + 1e-8)
    
    offsets = np.random.randn(num_control_points) * smoothness * length * 0.3
    control_points = base_points + offsets[:, None] * perp
    
    # Constrain to bounds if provided
    if bounds is not None:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        control_points[:, 0] = np.clip(control_points[:, 0], x_min, x_max)
        control_points[:, 1] = np.clip(control_points[:, 1], y_min, y_max)
    
    # Combine all control points
    all_points = [tuple(start)] + [tuple(p) for p in control_points] + [tuple(end)]
    
    return generate_bezier_trajectory(all_points, num_points=100)


# =============================================================================
# Trajectory Comparison
# =============================================================================

def frechet_distance_approx(traj1: Array, traj2: Array) -> float:
    """
    Approximate Fréchet distance between two trajectories.
    
    The Fréchet distance is the minimum length of a leash required to
    connect a dog and its owner as they walk along their respective paths.
    
    This is an approximation using discrete sampling.
    
    Args:
        traj1: First trajectory, shape (N, 2)
        traj2: Second trajectory, shape (M, 2)
        
    Returns:
        Approximate Fréchet distance
    """
    # Resample to same length
    n = max(traj1.shape[0], traj2.shape[0])
    t1 = resample_trajectory(traj1, n)
    t2 = resample_trajectory(traj2, n)
    
    # Compute pairwise distances
    dists = jnp.linalg.norm(t1 - t2, axis=1)
    
    # Fréchet distance is the maximum of minimum distances
    # This is a simplified approximation
    return jnp.max(dists)


def hausdorff_distance(traj1: Array, traj2: Array) -> float:
    """
    Compute Hausdorff distance between two trajectories.
    
    The Hausdorff distance is the maximum distance from any point on
    one trajectory to its nearest point on the other trajectory.
    
    Args:
        traj1: First trajectory, shape (N, 2)
        traj2: Second trajectory, shape (M, 2)
        
    Returns:
        Hausdorff distance
    """
    # Distance from each point in traj1 to nearest point in traj2
    def min_dist_to_traj(point, traj):
        dists = jnp.linalg.norm(traj - point, axis=1)
        return jnp.min(dists)
    
    # Max of min distances from traj1 to traj2
    dists_1_to_2 = jnp.array([min_dist_to_traj(p, traj2) for p in traj1])
    max_1_to_2 = jnp.max(dists_1_to_2)
    
    # Max of min distances from traj2 to traj1
    dists_2_to_1 = jnp.array([min_dist_to_traj(p, traj1) for p in traj2])
    max_2_to_1 = jnp.max(dists_2_to_1)
    
    return jnp.maximum(max_1_to_2, max_2_to_1)


# =============================================================================
# Trajectory Transformations
# =============================================================================

def translate_trajectory(trajectory: Array, offset: tuple[float, float]) -> Array:
    """Translate a trajectory by a fixed offset."""
    return trajectory + jnp.array(offset)


def scale_trajectory(
    trajectory: Array,
    scale: float | tuple[float, float],
    center: tuple[float, float] | None = None
) -> Array:
    """
    Scale a trajectory around a center point.
    
    Args:
        trajectory: Input trajectory
        scale: Scale factor (uniform) or (scale_x, scale_y)
        center: Center of scaling (default: trajectory centroid)
        
    Returns:
        Scaled trajectory
    """
    if center is None:
        center = trajectory_centroid(trajectory)
    else:
        center = jnp.array(center)
    
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    scale = jnp.array(scale)
    
    # Translate to origin, scale, translate back
    centered = trajectory - center
    scaled = centered * scale
    return scaled + center


def rotate_trajectory(
    trajectory: Array,
    angle: float,
    center: tuple[float, float] | None = None
) -> Array:
    """
    Rotate a trajectory around a center point.
    
    Args:
        trajectory: Input trajectory
        angle: Rotation angle in radians (CCW positive)
        center: Center of rotation (default: trajectory centroid)
        
    Returns:
        Rotated trajectory
    """
    if center is None:
        center = trajectory_centroid(trajectory)
    else:
        center = jnp.array(center)
    
    # Rotation matrix
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    rot_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Translate to origin, rotate, translate back
    centered = trajectory - center
    rotated = jnp.dot(centered, rot_matrix.T)
    return rotated + center


def reverse_trajectory(trajectory: Array) -> Array:
    """Reverse the direction of a trajectory."""
    return trajectory[::-1]

