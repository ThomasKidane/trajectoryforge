"""
Trajectory Encoder

Encodes variable-length trajectories into fixed-size latent representations.
Supports multiple architectures: CNN, Transformer, PointNet-style.
"""

from __future__ import annotations
from typing import Literal
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    d_model: int
    max_len: int = 1000
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        
        # Create position indices
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        
        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return x + pe


class CNNEncoder(nn.Module):
    """
    1D CNN encoder for trajectories.
    
    Treats trajectory as a 1D signal with 2 channels (x, y).
    """
    hidden_dim: int = 128
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, trajectory: Array, training: bool = True) -> Array:
        """
        Encode trajectory to latent vector.
        
        Args:
            trajectory: Shape (batch, seq_len, 2)
            training: Whether in training mode
            
        Returns:
            Latent vector, shape (batch, latent_dim)
        """
        x = trajectory
        
        # 1D convolutions
        x = nn.Conv(features=32, kernel_size=(5,))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5,))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.hidden_dim, kernel_size=(3,))(x)
        x = nn.relu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Final projection
        x = nn.Dense(self.latent_dim)(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for trajectories.
    
    Uses self-attention to capture long-range dependencies in the path.
    """
    hidden_dim: int = 128
    latent_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, trajectory: Array, training: bool = True) -> Array:
        """
        Encode trajectory to latent vector.
        
        Args:
            trajectory: Shape (batch, seq_len, 2)
            training: Whether in training mode
            
        Returns:
            Latent vector, shape (batch, latent_dim)
        """
        # Project input to hidden dimension
        x = nn.Dense(self.hidden_dim)(trajectory)
        
        # Add positional encoding
        x = PositionalEncoding(d_model=self.hidden_dim)(x)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Self-attention
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_dim,
            )(x, x)
            x = nn.LayerNorm()(x + attn_output)
            
            # Feed-forward
            ff_output = nn.Dense(self.hidden_dim * 4)(x)
            ff_output = nn.gelu(ff_output)
            ff_output = nn.Dense(self.hidden_dim)(ff_output)
            ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
            x = nn.LayerNorm()(x + ff_output)
        
        # Global average pooling (or use [CLS] token)
        x = jnp.mean(x, axis=1)
        
        # Final projection
        x = nn.Dense(self.latent_dim)(x)
        
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for trajectories.
    
    Processes each point independently, then aggregates with max pooling.
    Permutation invariant (order doesn't matter).
    """
    hidden_dim: int = 128
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, trajectory: Array, training: bool = True) -> Array:
        """
        Encode trajectory to latent vector.
        
        Args:
            trajectory: Shape (batch, seq_len, 2)
            training: Whether in training mode
            
        Returns:
            Latent vector, shape (batch, latent_dim)
        """
        x = trajectory
        
        # Per-point MLPs
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        
        # Max pooling over points (permutation invariant)
        x = jnp.max(x, axis=1)
        
        # Final projection
        x = nn.Dense(self.latent_dim)(x)
        
        return x


class TrajectoryEncoder(nn.Module):
    """
    Main trajectory encoder with selectable architecture.
    """
    architecture: Literal["cnn", "transformer", "pointnet"] = "transformer"
    hidden_dim: int = 128
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, trajectory: Array, training: bool = True) -> Array:
        """
        Encode trajectory to latent vector.
        
        Args:
            trajectory: Shape (batch, seq_len, 2) or (seq_len, 2)
            training: Whether in training mode
            
        Returns:
            Latent vector, shape (batch, latent_dim) or (latent_dim,)
        """
        # Handle unbatched input
        squeeze = False
        if trajectory.ndim == 2:
            trajectory = trajectory[None, ...]
            squeeze = True
        
        if self.architecture == "cnn":
            encoder = CNNEncoder(
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
        elif self.architecture == "transformer":
            encoder = TransformerEncoder(
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
        else:  # pointnet
            encoder = PointNetEncoder(
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
        
        latent = encoder(trajectory, training=training)
        
        if squeeze:
            latent = latent[0]
        
        return latent

