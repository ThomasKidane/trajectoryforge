"""
Force Field Decoder

Decodes latent representations into force field configurations.
Outputs parameters for a fixed maximum number of fields with validity masks.
"""

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
from flax import linen as nn
from jax import Array


class FieldPrediction(NamedTuple):
    """Predicted force field configuration."""
    # Field validity masks (which fields are active)
    wind_mask: Array      # (batch, max_wind)
    vortex_mask: Array    # (batch, max_vortex)
    point_mask: Array     # (batch, max_point)
    
    # Wind field parameters
    wind_centers: Array   # (batch, max_wind, 2)
    wind_sizes: Array     # (batch, max_wind, 2)
    wind_directions: Array  # (batch, max_wind, 2)
    wind_strengths: Array   # (batch, max_wind)
    
    # Vortex field parameters
    vortex_centers: Array   # (batch, max_vortex, 2)
    vortex_strengths: Array # (batch, max_vortex)
    vortex_radii: Array     # (batch, max_vortex)
    
    # Point force parameters
    point_centers: Array    # (batch, max_point, 2)
    point_strengths: Array  # (batch, max_point)


class ForceFieldDecoder(nn.Module):
    """
    Decode latent representation into force field parameters.
    
    Uses slot-based approach: predicts fixed max number of each field type,
    with validity masks to indicate which are active.
    """
    max_wind: int = 3
    max_vortex: int = 2
    max_point: int = 2
    hidden_dim: int = 128
    
    # Bounds for field parameters (for normalization)
    position_range: tuple[float, float] = (-5.0, 5.0)
    size_range: tuple[float, float] = (0.5, 3.0)
    strength_range: tuple[float, float] = (-20.0, 20.0)
    radius_range: tuple[float, float] = (0.5, 3.0)
    
    @nn.compact
    def __call__(self, latent: Array, training: bool = True) -> FieldPrediction:
        """
        Decode latent vector to field parameters.
        
        Args:
            latent: Shape (batch, latent_dim) or (latent_dim,)
            training: Whether in training mode
            
        Returns:
            FieldPrediction with all field parameters
        """
        # Handle unbatched input
        squeeze = False
        if latent.ndim == 1:
            latent = latent[None, ...]
            squeeze = True
        
        batch_size = latent.shape[0]
        
        # Shared MLP
        x = nn.Dense(self.hidden_dim)(latent)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # === Wind Fields ===
        wind_head = nn.Dense(self.hidden_dim)(x)
        wind_head = nn.relu(wind_head)
        
        # Validity mask (sigmoid for soft mask)
        wind_mask = nn.Dense(self.max_wind)(wind_head)
        wind_mask = jax.nn.sigmoid(wind_mask)
        
        # Centers (tanh scaled to position range)
        wind_centers = nn.Dense(self.max_wind * 2)(wind_head)
        wind_centers = wind_centers.reshape(batch_size, self.max_wind, 2)
        wind_centers = jax.nn.tanh(wind_centers) * (self.position_range[1] - self.position_range[0]) / 2
        
        # Sizes (sigmoid scaled to size range)
        wind_sizes = nn.Dense(self.max_wind * 2)(wind_head)
        wind_sizes = wind_sizes.reshape(batch_size, self.max_wind, 2)
        wind_sizes = jax.nn.sigmoid(wind_sizes) * (self.size_range[1] - self.size_range[0]) + self.size_range[0]
        
        # Directions (normalized)
        wind_directions = nn.Dense(self.max_wind * 2)(wind_head)
        wind_directions = wind_directions.reshape(batch_size, self.max_wind, 2)
        wind_directions = wind_directions / (jnp.linalg.norm(wind_directions, axis=-1, keepdims=True) + 1e-8)
        
        # Strengths
        wind_strengths = nn.Dense(self.max_wind)(wind_head)
        wind_strengths = jax.nn.tanh(wind_strengths) * self.strength_range[1]
        
        # === Vortex Fields ===
        if self.max_vortex > 0:
            vortex_head = nn.Dense(self.hidden_dim)(x)
            vortex_head = nn.relu(vortex_head)
            
            vortex_mask = nn.Dense(self.max_vortex)(vortex_head)
            vortex_mask = jax.nn.sigmoid(vortex_mask)
            
            vortex_centers = nn.Dense(self.max_vortex * 2)(vortex_head)
            vortex_centers = vortex_centers.reshape(batch_size, self.max_vortex, 2)
            vortex_centers = jax.nn.tanh(vortex_centers) * (self.position_range[1] - self.position_range[0]) / 2
            
            vortex_strengths = nn.Dense(self.max_vortex)(vortex_head)
            vortex_strengths = jax.nn.tanh(vortex_strengths) * self.strength_range[1]
            
            vortex_radii = nn.Dense(self.max_vortex)(vortex_head)
            vortex_radii = jax.nn.sigmoid(vortex_radii) * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
        else:
            vortex_mask = jnp.zeros((batch_size, 0))
            vortex_centers = jnp.zeros((batch_size, 0, 2))
            vortex_strengths = jnp.zeros((batch_size, 0))
            vortex_radii = jnp.zeros((batch_size, 0))
        
        # === Point Forces ===
        if self.max_point > 0:
            point_head = nn.Dense(self.hidden_dim)(x)
            point_head = nn.relu(point_head)
            
            point_mask = nn.Dense(self.max_point)(point_head)
            point_mask = jax.nn.sigmoid(point_mask)
            
            point_centers = nn.Dense(self.max_point * 2)(point_head)
            point_centers = point_centers.reshape(batch_size, self.max_point, 2)
            point_centers = jax.nn.tanh(point_centers) * (self.position_range[1] - self.position_range[0]) / 2
            
            point_strengths = nn.Dense(self.max_point)(point_head)
            point_strengths = jax.nn.tanh(point_strengths) * self.strength_range[1]
        else:
            point_mask = jnp.zeros((batch_size, 0))
            point_centers = jnp.zeros((batch_size, 0, 2))
            point_strengths = jnp.zeros((batch_size, 0))
        
        prediction = FieldPrediction(
            wind_mask=wind_mask,
            vortex_mask=vortex_mask,
            point_mask=point_mask,
            wind_centers=wind_centers,
            wind_sizes=wind_sizes,
            wind_directions=wind_directions,
            wind_strengths=wind_strengths,
            vortex_centers=vortex_centers,
            vortex_strengths=vortex_strengths,
            vortex_radii=vortex_radii,
            point_centers=point_centers,
            point_strengths=point_strengths,
        )
        
        if squeeze:
            # Remove batch dimension
            prediction = FieldPrediction(
                wind_mask=prediction.wind_mask[0],
                vortex_mask=prediction.vortex_mask[0],
                point_mask=prediction.point_mask[0],
                wind_centers=prediction.wind_centers[0],
                wind_sizes=prediction.wind_sizes[0],
                wind_directions=prediction.wind_directions[0],
                wind_strengths=prediction.wind_strengths[0],
                vortex_centers=prediction.vortex_centers[0],
                vortex_strengths=prediction.vortex_strengths[0],
                vortex_radii=prediction.vortex_radii[0],
                point_centers=prediction.point_centers[0],
                point_strengths=prediction.point_strengths[0],
            )
        
        return prediction


# Import jax for sigmoid
import jax

