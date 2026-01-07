"""
Force Field Implementations

This module defines various force fields that can be placed in the simulation.
All force computations are fully differentiable with respect to field parameters.

Force Fields:
- WindField: Constant directional force in a rectangular region
- VortexField: Tangential force creating rotation
- PointForce: Inverse-square attractor/repeller
- GravityWell: Localized gravitational field
- DragZone: Velocity-dependent damping region
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple, Literal
import jax
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Base Force Field
# =============================================================================

class ForceField(ABC):
    """
    Abstract base class for all force fields.
    
    All subclasses must implement compute_force() which returns the force
    vector acting on a particle at a given position with given velocity.
    
    The implementation must be fully JAX-differentiable.
    """
    
    @abstractmethod
    def compute_force(self, position: Array, velocity: Array) -> Array:
        """
        Compute the force exerted by this field on a particle.
        
        Args:
            position: Particle position, shape (2,)
            velocity: Particle velocity, shape (2,)
            
        Returns:
            Force vector, shape (2,)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Return field parameters as a dictionary for serialization."""
        pass
    
    @classmethod
    @abstractmethod
    def from_params(cls, params: dict) -> ForceField:
        """Create a field from a parameter dictionary."""
        pass


# =============================================================================
# Utility Functions for Smooth/Soft Operations
# =============================================================================

def soft_indicator_box(
    position: Array,
    center: Array,
    half_size: Array,
    softness: float = 0.1
) -> float:
    """
    Soft indicator function for a rectangular region.
    
    Returns ~1 inside the box, ~0 outside, with smooth transition.
    Uses product of sigmoids for each dimension.
    
    Args:
        position: Query position, shape (2,)
        center: Box center, shape (2,)
        half_size: Half-width and half-height, shape (2,)
        softness: Transition width (smaller = sharper)
        
    Returns:
        Soft indicator value in [0, 1]
    """
    # Distance from center in each dimension, normalized by half_size
    rel_pos = (position - center) / half_size
    
    # Soft indicator: product of (1 - |x|)_+ for each dimension
    # Using sigmoid for smooth approximation
    scale = 1.0 / softness
    
    # For each dimension, we want high value when |rel_pos| < 1
    indicators = jax.nn.sigmoid(scale * (1.0 - jnp.abs(rel_pos)))
    
    return jnp.prod(indicators)


def soft_indicator_circle(
    position: Array,
    center: Array,
    radius: float,
    softness: float = 0.1
) -> float:
    """
    Soft indicator function for a circular region.
    
    Returns ~1 inside the circle, ~0 outside, with smooth transition.
    
    Args:
        position: Query position, shape (2,)
        center: Circle center, shape (2,)
        radius: Circle radius
        softness: Transition width (smaller = sharper)
        
    Returns:
        Soft indicator value in [0, 1]
    """
    distance = jnp.linalg.norm(position - center)
    # Sigmoid centered at radius
    return jax.nn.sigmoid((radius - distance) / softness)


def safe_normalize(vec: Array, eps: float = 1e-8) -> Array:
    """
    Safely normalize a vector, handling zero-length case.
    
    Args:
        vec: Input vector, shape (2,)
        eps: Small constant to avoid division by zero
        
    Returns:
        Normalized vector, shape (2,)
    """
    norm = jnp.linalg.norm(vec)
    return vec / (norm + eps)


def safe_inverse_power(distance: float, power: float, eps: float = 1e-6) -> float:
    """
    Safely compute 1/r^power, avoiding singularity at r=0.
    
    Uses softened formula: 1 / (r^2 + eps)^(power/2)
    
    Args:
        distance: Distance value
        power: Power for inverse law (e.g., 2 for inverse-square)
        eps: Softening parameter
        
    Returns:
        Softened inverse power value
    """
    return 1.0 / (distance ** 2 + eps) ** (power / 2)


# =============================================================================
# Wind Field
# =============================================================================

class WindField(NamedTuple):
    """
    Uniform wind force in a rectangular region.
    
    Creates a constant directional force within a box-shaped area.
    The force smoothly falls off at the boundaries for differentiability.
    
    Attributes:
        center: Center position of the wind region, shape (2,)
        size: Width and height of the region, shape (2,)
        direction: Force direction (will be normalized), shape (2,)
        strength: Magnitude of the force
        softness: Boundary transition width
    """
    center: Array       # Shape: (2,)
    size: Array         # Shape: (2,) - full width and height
    direction: Array    # Shape: (2,) - force direction
    strength: float
    softness: float = 0.1
    
    @staticmethod
    def create(
        center: tuple[float, float],
        size: tuple[float, float],
        direction: tuple[float, float],
        strength: float,
        softness: float = 0.1
    ) -> WindField:
        """Create a WindField from Python tuples."""
        dir_array = jnp.array(direction, dtype=jnp.float32)
        # Normalize direction
        dir_norm = dir_array / (jnp.linalg.norm(dir_array) + 1e-8)
        return WindField(
            center=jnp.array(center, dtype=jnp.float32),
            size=jnp.array(size, dtype=jnp.float32),
            direction=dir_norm,
            strength=float(strength),
            softness=float(softness)
        )
    
    def compute_force(self, position: Array, velocity: Array) -> Array:
        """Compute wind force at given position."""
        # Soft indicator for being inside the wind region
        half_size = self.size / 2.0
        indicator = soft_indicator_box(position, self.center, half_size, self.softness)
        
        # Force = strength * direction * indicator
        return self.strength * self.direction * indicator
    
    def get_params(self) -> dict:
        return {
            "type": "wind",
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "direction": self.direction.tolist(),
            "strength": self.strength,
            "softness": self.softness
        }
    
    @classmethod
    def from_params(cls, params: dict) -> WindField:
        return WindField.create(
            center=tuple(params["center"]),
            size=tuple(params["size"]),
            direction=tuple(params["direction"]),
            strength=params["strength"],
            softness=params.get("softness", 0.1)
        )


# =============================================================================
# Vortex Field
# =============================================================================

class VortexField(NamedTuple):
    """
    Vortex/tornado force field creating rotation.
    
    Creates a tangential force that causes particles to rotate around
    the vortex center. The force can fall off with distance using
    different profiles.
    
    Attributes:
        center: Vortex center position, shape (2,)
        strength: Rotation strength (positive = CCW, negative = CW)
        radius: Characteristic radius for falloff
        falloff: Type of distance falloff ('constant', 'linear', 'inverse', 'gaussian')
        softness: Core softening to avoid singularity
    """
    center: Array
    strength: float
    radius: float
    falloff: str = "gaussian"  # 'constant', 'linear', 'inverse', 'gaussian'
    softness: float = 0.1
    
    @staticmethod
    def create(
        center: tuple[float, float],
        strength: float,
        radius: float,
        falloff: str = "gaussian",
        softness: float = 0.1
    ) -> VortexField:
        """Create a VortexField from Python tuples."""
        return VortexField(
            center=jnp.array(center, dtype=jnp.float32),
            strength=float(strength),
            radius=float(radius),
            falloff=falloff,
            softness=float(softness)
        )
    
    def compute_force(self, position: Array, velocity: Array) -> Array:
        """Compute vortex force at given position."""
        # Vector from center to position
        r_vec = position - self.center
        distance = jnp.linalg.norm(r_vec)
        
        # Tangential direction (perpendicular to radial)
        # For CCW rotation: (-y, x) / |r|
        tangent = jnp.array([-r_vec[1], r_vec[0]]) / (distance + self.softness)
        
        # Compute falloff based on type
        normalized_dist = distance / self.radius
        
        # Using jnp.where for differentiable branching
        falloff_constant = jnp.ones_like(distance)
        falloff_linear = jnp.maximum(0.0, 1.0 - normalized_dist)
        falloff_inverse = 1.0 / (1.0 + normalized_dist)
        falloff_gaussian = jnp.exp(-0.5 * normalized_dist ** 2)
        
        # Select falloff (this is a bit hacky but keeps things differentiable)
        # In practice, you'd use a single falloff type per field
        falloff_value = jnp.where(
            self.falloff == "constant", falloff_constant,
            jnp.where(
                self.falloff == "linear", falloff_linear,
                jnp.where(
                    self.falloff == "inverse", falloff_inverse,
                    falloff_gaussian  # default
                )
            )
        )
        
        return self.strength * tangent * falloff_value
    
    def get_params(self) -> dict:
        return {
            "type": "vortex",
            "center": self.center.tolist(),
            "strength": self.strength,
            "radius": self.radius,
            "falloff": self.falloff,
            "softness": self.softness
        }
    
    @classmethod
    def from_params(cls, params: dict) -> VortexField:
        return VortexField.create(
            center=tuple(params["center"]),
            strength=params["strength"],
            radius=params["radius"],
            falloff=params.get("falloff", "gaussian"),
            softness=params.get("softness", 0.1)
        )


# =============================================================================
# Point Force (Attractor/Repeller)
# =============================================================================

class PointForce(NamedTuple):
    """
    Point attractor or repeller with inverse-power falloff.
    
    Creates a radial force toward (attractor) or away from (repeller)
    a point. Similar to electrostatic or gravitational point sources.
    
    Attributes:
        center: Force center position, shape (2,)
        strength: Force magnitude (positive = attractor, negative = repeller)
        falloff_power: Exponent for inverse-power law (2.0 = inverse-square)
        max_distance: Maximum effective distance (force = 0 beyond this)
        softness: Core softening to avoid singularity at center
    """
    center: Array
    strength: float
    falloff_power: float = 2.0
    max_distance: float = 10.0
    softness: float = 0.1
    
    @staticmethod
    def create(
        center: tuple[float, float],
        strength: float,
        falloff_power: float = 2.0,
        max_distance: float = 10.0,
        softness: float = 0.1
    ) -> PointForce:
        """Create a PointForce from Python tuples."""
        return PointForce(
            center=jnp.array(center, dtype=jnp.float32),
            strength=float(strength),
            falloff_power=float(falloff_power),
            max_distance=float(max_distance),
            softness=float(softness)
        )
    
    def compute_force(self, position: Array, velocity: Array) -> Array:
        """Compute point force at given position."""
        # Vector from position to center (for attractor)
        r_vec = self.center - position
        distance = jnp.linalg.norm(r_vec)
        
        # Normalized direction toward center
        direction = r_vec / (distance + 1e-8)
        
        # Inverse-power magnitude with softening
        # F = strength / (r^2 + softness^2)^(power/2)
        magnitude = self.strength * safe_inverse_power(
            distance, self.falloff_power, self.softness ** 2
        )
        
        # Smooth cutoff at max_distance
        cutoff = jax.nn.sigmoid((self.max_distance - distance) / (self.softness * 2))
        
        return magnitude * direction * cutoff
    
    def get_params(self) -> dict:
        return {
            "type": "point",
            "center": self.center.tolist(),
            "strength": self.strength,
            "falloff_power": self.falloff_power,
            "max_distance": self.max_distance,
            "softness": self.softness
        }
    
    @classmethod
    def from_params(cls, params: dict) -> PointForce:
        return PointForce.create(
            center=tuple(params["center"]),
            strength=params["strength"],
            falloff_power=params.get("falloff_power", 2.0),
            max_distance=params.get("max_distance", 10.0),
            softness=params.get("softness", 0.1)
        )


# =============================================================================
# Gravity Well
# =============================================================================

class GravityWell(NamedTuple):
    """
    Localized gravitational field with Gaussian profile.
    
    Unlike PointForce which has inverse-square falloff, GravityWell
    uses a Gaussian profile that's stronger near the center and
    smoothly falls off. This creates more predictable behavior.
    
    Attributes:
        center: Well center position, shape (2,)
        strength: Gravitational strength (positive = attractive)
        sigma: Gaussian width parameter
        max_force: Maximum force magnitude (clamping for stability)
    """
    center: Array
    strength: float
    sigma: float = 1.0
    max_force: float = 50.0
    
    @staticmethod
    def create(
        center: tuple[float, float],
        strength: float,
        sigma: float = 1.0,
        max_force: float = 50.0
    ) -> GravityWell:
        """Create a GravityWell from Python tuples."""
        return GravityWell(
            center=jnp.array(center, dtype=jnp.float32),
            strength=float(strength),
            sigma=float(sigma),
            max_force=float(max_force)
        )
    
    def compute_force(self, position: Array, velocity: Array) -> Array:
        """Compute gravity well force at given position."""
        # Vector from position to center
        r_vec = self.center - position
        distance = jnp.linalg.norm(r_vec)
        
        # Normalized direction
        direction = r_vec / (distance + 1e-8)
        
        # Gaussian profile: force increases toward center, then Gaussian falloff
        # F(r) = strength * r * exp(-r^2 / (2*sigma^2)) / sigma^2
        # This gives zero force at center, peaks at r=sigma, then falls off
        gaussian_factor = jnp.exp(-0.5 * (distance / self.sigma) ** 2)
        magnitude = self.strength * (distance / self.sigma) * gaussian_factor
        
        # Clamp to max_force for stability
        magnitude = jnp.clip(magnitude, -self.max_force, self.max_force)
        
        return magnitude * direction
    
    def get_params(self) -> dict:
        return {
            "type": "gravity_well",
            "center": self.center.tolist(),
            "strength": self.strength,
            "sigma": self.sigma,
            "max_force": self.max_force
        }
    
    @classmethod
    def from_params(cls, params: dict) -> GravityWell:
        return GravityWell.create(
            center=tuple(params["center"]),
            strength=params["strength"],
            sigma=params.get("sigma", 1.0),
            max_force=params.get("max_force", 50.0)
        )


# =============================================================================
# Drag Zone
# =============================================================================

class DragZone(NamedTuple):
    """
    Velocity-dependent damping region.
    
    Creates a drag force proportional to velocity within a circular region.
    Useful for slowing down particles in specific areas.
    
    F = -drag_coefficient * velocity * indicator(position)
    
    Attributes:
        center: Zone center position, shape (2,)
        radius: Zone radius
        drag_coefficient: Drag strength (higher = more damping)
        softness: Boundary transition width
    """
    center: Array
    radius: float
    drag_coefficient: float
    softness: float = 0.1
    
    @staticmethod
    def create(
        center: tuple[float, float],
        radius: float,
        drag_coefficient: float,
        softness: float = 0.1
    ) -> DragZone:
        """Create a DragZone from Python tuples."""
        return DragZone(
            center=jnp.array(center, dtype=jnp.float32),
            radius=float(radius),
            drag_coefficient=float(drag_coefficient),
            softness=float(softness)
        )
    
    def compute_force(self, position: Array, velocity: Array) -> Array:
        """Compute drag force at given position with given velocity."""
        # Soft indicator for being inside the drag zone
        indicator = soft_indicator_circle(position, self.center, self.radius, self.softness)
        
        # Drag force opposes velocity
        return -self.drag_coefficient * velocity * indicator
    
    def get_params(self) -> dict:
        return {
            "type": "drag",
            "center": self.center.tolist(),
            "radius": self.radius,
            "drag_coefficient": self.drag_coefficient,
            "softness": self.softness
        }
    
    @classmethod
    def from_params(cls, params: dict) -> DragZone:
        return DragZone.create(
            center=tuple(params["center"]),
            radius=params["radius"],
            drag_coefficient=params["drag_coefficient"],
            softness=params.get("softness", 0.1)
        )


# =============================================================================
# Field Collection Utilities
# =============================================================================

# Type for any force field
AnyField = WindField | VortexField | PointForce | GravityWell | DragZone


def compute_field_force(field: AnyField, position: Array, velocity: Array) -> Array:
    """
    Compute force from any field type.
    
    This is a dispatcher function that works with all field types.
    """
    return field.compute_force(position, velocity)


def field_from_params(params: dict) -> AnyField:
    """
    Create a force field from a parameter dictionary.
    
    The dictionary must have a "type" key indicating the field type.
    """
    field_type = params["type"]
    
    if field_type == "wind":
        return WindField.from_params(params)
    elif field_type == "vortex":
        return VortexField.from_params(params)
    elif field_type == "point":
        return PointForce.from_params(params)
    elif field_type == "gravity_well":
        return GravityWell.from_params(params)
    elif field_type == "drag":
        return DragZone.from_params(params)
    else:
        raise ValueError(f"Unknown field type: {field_type}")


def fields_to_params(fields: list[AnyField]) -> list[dict]:
    """Convert a list of fields to parameter dictionaries."""
    return [f.get_params() for f in fields]


def params_to_fields(params_list: list[dict]) -> list[AnyField]:
    """Create fields from a list of parameter dictionaries."""
    return [field_from_params(p) for p in params_list]

