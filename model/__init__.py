"""
TrajectoryForge Model Module

Neural network architectures for solving the inverse problem:
trajectory → force field configuration
"""

from model.encoder import TrajectoryEncoder
from model.decoder import ForceFieldDecoder
from model.model import InverseModel, create_model
from model.training import (
    Trainer,
    TrainingConfig,
    generate_training_sample,
    generate_training_dataset,
    evaluate_model,
)
from model.curriculum import (
    DifficultyLevel,
    DIFFICULTY_LEVELS,
    CurriculumScheduler,
    CurriculumTrainer,
    generate_sample_at_difficulty,
    generate_curriculum_dataset,
)
from model.augmentation import (
    AugmentationPipeline,
    create_strong_augmentation,
    create_light_augmentation,
    augment_sample,
    augment_dataset,
)
from model.scheduler import (
    warmup_cosine_decay_schedule,
    warmup_linear_decay_schedule,
    one_cycle_schedule,
    create_optimizer,
    create_adam_with_warmup,
)
from model.architectures import (
    ModelConfig,
    MODEL_CONFIGS,
    create_model_from_config,
    get_config,
    list_configs,
    describe_configs,
    ModelConfigBuilder,
)

__all__ = [
    # Core model
    "TrajectoryEncoder",
    "ForceFieldDecoder", 
    "InverseModel",
    "create_model",
    # Training
    "Trainer",
    "TrainingConfig",
    "generate_training_sample",
    "generate_training_dataset",
    "evaluate_model",
    # Curriculum learning
    "DifficultyLevel",
    "DIFFICULTY_LEVELS",
    "CurriculumScheduler",
    "CurriculumTrainer",
    "generate_sample_at_difficulty",
    "generate_curriculum_dataset",
    # Augmentation
    "AugmentationPipeline",
    "create_strong_augmentation",
    "create_light_augmentation",
    "augment_sample",
    "augment_dataset",
    # Scheduling
    "warmup_cosine_decay_schedule",
    "warmup_linear_decay_schedule",
    "one_cycle_schedule",
    "create_optimizer",
    "create_adam_with_warmup",
    # Architectures
    "ModelConfig",
    "MODEL_CONFIGS",
    "create_model_from_config",
    "get_config",
    "list_configs",
    "describe_configs",
    "ModelConfigBuilder",
]

