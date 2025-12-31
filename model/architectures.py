"""
Model Architecture Variants

Provides different model configurations for experimentation,
from small/fast models to large/powerful ones.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from model.model import InverseModel, create_model


# =============================================================================
# Model Configurations
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model variant."""
    name: str
    encoder_type: Literal["cnn", "transformer", "pointnet"]
    hidden_dim: int
    latent_dim: int
    max_wind: int = 1
    max_vortex: int = 0
    max_point: int = 0
    description: str = ""
    
    # Estimated metrics
    approx_params: str = ""
    training_speed: str = ""  # relative speed
    
    def create(self) -> InverseModel:
        """Create the model from this configuration."""
        return create_model(
            encoder_type=self.encoder_type,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            max_wind=self.max_wind,
            max_vortex=self.max_vortex,
            max_point=self.max_point,
        )


# =============================================================================
# Predefined Configurations
# =============================================================================

# Tiny model for quick experiments
TINY_CONFIG = ModelConfig(
    name="tiny",
    encoder_type="cnn",
    hidden_dim=32,
    latent_dim=16,
    max_wind=1,
    max_vortex=0,
    max_point=0,
    description="Minimal model for quick testing and debugging",
    approx_params="~5K",
    training_speed="very fast",
)

# Small model - good for simple trajectories
SMALL_CONFIG = ModelConfig(
    name="small",
    encoder_type="cnn",
    hidden_dim=64,
    latent_dim=32,
    max_wind=1,
    max_vortex=0,
    max_point=0,
    description="Small CNN model for simple wind-only trajectories",
    approx_params="~20K",
    training_speed="fast",
)

# Medium model - balanced performance
MEDIUM_CONFIG = ModelConfig(
    name="medium",
    encoder_type="transformer",
    hidden_dim=128,
    latent_dim=64,
    max_wind=2,
    max_vortex=1,
    max_point=0,
    description="Balanced transformer model for mixed wind/vortex fields",
    approx_params="~100K",
    training_speed="medium",
)

# Large model - high capacity
LARGE_CONFIG = ModelConfig(
    name="large",
    encoder_type="transformer",
    hidden_dim=256,
    latent_dim=128,
    max_wind=3,
    max_vortex=2,
    max_point=2,
    description="Large transformer for complex multi-field configurations",
    approx_params="~500K",
    training_speed="slow",
)

# XLarge model - maximum capacity
XLARGE_CONFIG = ModelConfig(
    name="xlarge",
    encoder_type="transformer",
    hidden_dim=512,
    latent_dim=256,
    max_wind=4,
    max_vortex=3,
    max_point=3,
    description="Extra large model for the most complex scenarios",
    approx_params="~2M",
    training_speed="very slow",
)

# PointNet variant - permutation invariant
POINTNET_CONFIG = ModelConfig(
    name="pointnet",
    encoder_type="pointnet",
    hidden_dim=128,
    latent_dim=64,
    max_wind=2,
    max_vortex=1,
    max_point=1,
    description="PointNet encoder for unordered trajectory points",
    approx_params="~80K",
    training_speed="medium",
)

# CNN variant - fast inference
CNN_FAST_CONFIG = ModelConfig(
    name="cnn_fast",
    encoder_type="cnn",
    hidden_dim=128,
    latent_dim=64,
    max_wind=2,
    max_vortex=1,
    max_point=0,
    description="CNN encoder optimized for fast inference",
    approx_params="~60K",
    training_speed="fast",
)

# Wind-only specialist
WIND_SPECIALIST_CONFIG = ModelConfig(
    name="wind_specialist",
    encoder_type="transformer",
    hidden_dim=192,
    latent_dim=96,
    max_wind=4,
    max_vortex=0,
    max_point=0,
    description="Specialized for wind-field-only scenarios",
    approx_params="~200K",
    training_speed="medium",
)

# All configs in a dict for easy access
MODEL_CONFIGS = {
    "tiny": TINY_CONFIG,
    "small": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
    "large": LARGE_CONFIG,
    "xlarge": XLARGE_CONFIG,
    "pointnet": POINTNET_CONFIG,
    "cnn_fast": CNN_FAST_CONFIG,
    "wind_specialist": WIND_SPECIALIST_CONFIG,
}


# =============================================================================
# Model Factory
# =============================================================================

def create_model_from_config(config_name: str) -> InverseModel:
    """
    Create a model from a predefined configuration name.
    
    Args:
        config_name: Name of the configuration (tiny, small, medium, large, xlarge)
        
    Returns:
        Configured InverseModel
    """
    if config_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    
    return MODEL_CONFIGS[config_name].create()


def get_config(config_name: str) -> ModelConfig:
    """Get a model configuration by name."""
    if config_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    
    return MODEL_CONFIGS[config_name]


def list_configs() -> list[str]:
    """List all available configuration names."""
    return list(MODEL_CONFIGS.keys())


def describe_configs() -> str:
    """Get a formatted description of all configurations."""
    lines = ["Available Model Configurations:", "=" * 40]
    
    for name, config in MODEL_CONFIGS.items():
        lines.append(f"\n{name.upper()}")
        lines.append(f"  Encoder: {config.encoder_type}")
        lines.append(f"  Hidden dim: {config.hidden_dim}")
        lines.append(f"  Latent dim: {config.latent_dim}")
        lines.append(f"  Max fields: wind={config.max_wind}, vortex={config.max_vortex}, point={config.max_point}")
        lines.append(f"  Approx params: {config.approx_params}")
        lines.append(f"  Training speed: {config.training_speed}")
        lines.append(f"  {config.description}")
    
    return "\n".join(lines)


# =============================================================================
# Custom Configuration Builder
# =============================================================================

class ModelConfigBuilder:
    """Builder for creating custom model configurations."""
    
    def __init__(self, base_config: str = "medium"):
        """
        Initialize builder from a base configuration.
        
        Args:
            base_config: Name of base config to start from
        """
        base = MODEL_CONFIGS[base_config]
        self._name = f"custom_{base_config}"
        self._encoder_type = base.encoder_type
        self._hidden_dim = base.hidden_dim
        self._latent_dim = base.latent_dim
        self._max_wind = base.max_wind
        self._max_vortex = base.max_vortex
        self._max_point = base.max_point
        self._description = "Custom configuration"
    
    def name(self, name: str) -> "ModelConfigBuilder":
        """Set configuration name."""
        self._name = name
        return self
    
    def encoder(self, encoder_type: str) -> "ModelConfigBuilder":
        """Set encoder type."""
        self._encoder_type = encoder_type
        return self
    
    def hidden_dim(self, dim: int) -> "ModelConfigBuilder":
        """Set hidden dimension."""
        self._hidden_dim = dim
        return self
    
    def latent_dim(self, dim: int) -> "ModelConfigBuilder":
        """Set latent dimension."""
        self._latent_dim = dim
        return self
    
    def max_fields(
        self,
        wind: int = None,
        vortex: int = None,
        point: int = None,
    ) -> "ModelConfigBuilder":
        """Set maximum number of each field type."""
        if wind is not None:
            self._max_wind = wind
        if vortex is not None:
            self._max_vortex = vortex
        if point is not None:
            self._max_point = point
        return self
    
    def description(self, desc: str) -> "ModelConfigBuilder":
        """Set configuration description."""
        self._description = desc
        return self
    
    def scale(self, factor: float) -> "ModelConfigBuilder":
        """
        Scale hidden and latent dimensions by a factor.
        
        Args:
            factor: Scaling factor (e.g., 2.0 doubles dimensions)
        """
        self._hidden_dim = int(self._hidden_dim * factor)
        self._latent_dim = int(self._latent_dim * factor)
        return self
    
    def build(self) -> ModelConfig:
        """Build the configuration."""
        return ModelConfig(
            name=self._name,
            encoder_type=self._encoder_type,
            hidden_dim=self._hidden_dim,
            latent_dim=self._latent_dim,
            max_wind=self._max_wind,
            max_vortex=self._max_vortex,
            max_point=self._max_point,
            description=self._description,
        )
    
    def create(self) -> InverseModel:
        """Build configuration and create the model."""
        return self.build().create()


# =============================================================================
# Ensemble Models
# =============================================================================

def create_ensemble(
    configs: list[str] = None,
    num_models: int = 3,
) -> list[InverseModel]:
    """
    Create an ensemble of models with different configurations.
    
    Args:
        configs: List of config names to use. If None, uses diverse defaults.
        num_models: Number of models if configs is None
        
    Returns:
        List of InverseModel instances
    """
    if configs is None:
        # Default diverse ensemble
        configs = ["small", "medium", "pointnet"][:num_models]
    
    return [create_model_from_config(c) for c in configs]


# =============================================================================
# Architecture Search Space (for hyperparameter tuning)
# =============================================================================

ARCHITECTURE_SEARCH_SPACE = {
    "encoder_type": ["cnn", "transformer", "pointnet"],
    "hidden_dim": [32, 64, 128, 256, 512],
    "latent_dim": [16, 32, 64, 128, 256],
    "max_wind": [1, 2, 3, 4],
    "max_vortex": [0, 1, 2, 3],
    "max_point": [0, 1, 2],
}


def sample_architecture(key) -> ModelConfig:
    """
    Sample a random architecture from the search space.
    
    Args:
        key: JAX random key
        
    Returns:
        Randomly sampled ModelConfig
    """
    import jax.random as random
    
    keys = random.split(key, 6)
    
    encoder_idx = random.randint(keys[0], (), 0, len(ARCHITECTURE_SEARCH_SPACE["encoder_type"]))
    hidden_idx = random.randint(keys[1], (), 0, len(ARCHITECTURE_SEARCH_SPACE["hidden_dim"]))
    latent_idx = random.randint(keys[2], (), 0, len(ARCHITECTURE_SEARCH_SPACE["latent_dim"]))
    wind_idx = random.randint(keys[3], (), 0, len(ARCHITECTURE_SEARCH_SPACE["max_wind"]))
    vortex_idx = random.randint(keys[4], (), 0, len(ARCHITECTURE_SEARCH_SPACE["max_vortex"]))
    point_idx = random.randint(keys[5], (), 0, len(ARCHITECTURE_SEARCH_SPACE["max_point"]))
    
    return ModelConfig(
        name="sampled",
        encoder_type=ARCHITECTURE_SEARCH_SPACE["encoder_type"][int(encoder_idx)],
        hidden_dim=ARCHITECTURE_SEARCH_SPACE["hidden_dim"][int(hidden_idx)],
        latent_dim=ARCHITECTURE_SEARCH_SPACE["latent_dim"][int(latent_idx)],
        max_wind=ARCHITECTURE_SEARCH_SPACE["max_wind"][int(wind_idx)],
        max_vortex=ARCHITECTURE_SEARCH_SPACE["max_vortex"][int(vortex_idx)],
        max_point=ARCHITECTURE_SEARCH_SPACE["max_point"][int(point_idx)],
        description="Randomly sampled architecture",
    )

