"""
TrajectoryForge Data Module

Data generation, datasets, and augmentation for training.
"""

from data.generator import (
    generate_random_fields,
    generate_trajectory_from_fields,
    generate_dataset_sample,
)
from data.dataset import TrajectoryDataset
from data.augmentation import augment_trajectory

__all__ = [
    "generate_random_fields",
    "generate_trajectory_from_fields", 
    "generate_dataset_sample",
    "TrajectoryDataset",
    "augment_trajectory",
]

