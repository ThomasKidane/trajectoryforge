"""
Inverse Model

Full model that maps trajectories to force field configurations.
Combines encoder and decoder with end-to-end training through physics.
"""

from __future__ import annotations
from typing import Literal
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from model.encoder import TrajectoryEncoder
from model.decoder import ForceFieldDecoder, FieldPrediction


class InverseModel(nn.Module):
    """
    Full inverse model: trajectory → force field configuration.
    
    This model solves the inverse problem by learning to predict
    force field parameters that would produce a given trajectory.
    """
    # Encoder config
    encoder_architecture: Literal["cnn", "transformer", "pointnet"] = "transformer"
    hidden_dim: int = 128
    latent_dim: int = 64
    
    # Decoder config
    max_wind: int = 3
    max_vortex: int = 2
    max_point: int = 2
    
    @nn.compact
    def __call__(
        self, 
        trajectory: Array, 
        training: bool = True
    ) -> FieldPrediction:
        """
        Predict force field configuration from trajectory.
        
        Args:
            trajectory: Target trajectory, shape (batch, seq_len, 2) or (seq_len, 2)
            training: Whether in training mode
            
        Returns:
            FieldPrediction with predicted field parameters
        """
        # Encode trajectory to latent
        encoder = TrajectoryEncoder(
            architecture=self.encoder_architecture,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        latent = encoder(trajectory, training=training)
        
        # Decode latent to field parameters
        decoder = ForceFieldDecoder(
            max_wind=self.max_wind,
            max_vortex=self.max_vortex,
            max_point=self.max_point,
            hidden_dim=self.hidden_dim,
        )
        prediction = decoder(latent, training=training)
        
        return prediction


def prediction_to_fields(prediction: FieldPrediction, threshold: float = 0.5):
    """
    Convert model prediction to actual force field objects.
    
    Args:
        prediction: FieldPrediction from the model
        threshold: Mask threshold for considering a field active
        
    Returns:
        List of force field objects
    """
    from physics.fields import WindField, VortexField, PointForce
    
    fields = []
    
    # Wind fields
    for i in range(len(prediction.wind_mask)):
        if prediction.wind_mask[i] > threshold:
            field = WindField(
                center=prediction.wind_centers[i],
                size=prediction.wind_sizes[i],
                direction=prediction.wind_directions[i],
                strength=float(prediction.wind_strengths[i]),
                softness=0.1,
            )
            fields.append(field)
    
    # Vortex fields
    for i in range(len(prediction.vortex_mask)):
        if prediction.vortex_mask[i] > threshold:
            field = VortexField(
                center=prediction.vortex_centers[i],
                strength=float(prediction.vortex_strengths[i]),
                radius=float(prediction.vortex_radii[i]),
                falloff="gaussian",
                softness=0.1,
            )
            fields.append(field)
    
    # Point forces
    for i in range(len(prediction.point_mask)):
        if prediction.point_mask[i] > threshold:
            field = PointForce(
                center=prediction.point_centers[i],
                strength=float(prediction.point_strengths[i]),
                falloff_power=2.0,
                max_distance=10.0,
                softness=0.1,
            )
            fields.append(field)
    
    return fields


def create_model(
    encoder_type: str = "transformer",
    hidden_dim: int = 128,
    latent_dim: int = 64,
    max_wind: int = 3,
    max_vortex: int = 2,
    max_point: int = 2,
) -> InverseModel:
    """
    Factory function to create an InverseModel.
    
    Args:
        encoder_type: Type of trajectory encoder
        hidden_dim: Hidden dimension for networks
        latent_dim: Latent space dimension
        max_wind: Maximum number of wind fields
        max_vortex: Maximum number of vortex fields
        max_point: Maximum number of point forces
        
    Returns:
        Configured InverseModel
    """
    return InverseModel(
        encoder_architecture=encoder_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_wind=max_wind,
        max_vortex=max_vortex,
        max_point=max_point,
    )

