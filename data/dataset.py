"""
Trajectory Dataset

PyTorch-compatible dataset for training the inverse model.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
import jax.numpy as jnp
from jax import random

from data.generator import generate_dataset_sample
from physics.state import SimulationConfig


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory-to-field training.
    
    Can either load pre-generated data or generate on-the-fly.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        num_wind: int = 1,
        num_vortex: int = 1,
        num_point: int = 0,
        trajectory_length: int = 300,
        seed: int = 42,
        pregenerate: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            num_samples: Number of samples in dataset
            num_wind: Number of wind fields per sample
            num_vortex: Number of vortex fields per sample
            num_point: Number of point forces per sample
            trajectory_length: Number of points in each trajectory
            seed: Random seed
            pregenerate: If True, generate all data upfront
        """
        self.num_samples = num_samples
        self.num_wind = num_wind
        self.num_vortex = num_vortex
        self.num_point = num_point
        self.trajectory_length = trajectory_length
        self.seed = seed
        
        self.config = SimulationConfig(
            dt=0.01,
            num_steps=trajectory_length,
            bounds=((-5.0, 5.0), (-5.0, 5.0)),
        )
        
        self.data = None
        if pregenerate:
            self._pregenerate()
    
    def _pregenerate(self):
        """Pre-generate all dataset samples."""
        from data.generator import generate_dataset
        
        samples = generate_dataset(
            num_samples=self.num_samples,
            seed=self.seed,
            num_wind=self.num_wind,
            num_vortex=self.num_vortex,
            num_point=self.num_point,
            config=self.config,
        )
        
        self.data = samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - trajectory: (trajectory_length, 2) tensor
                - initial_position: (2,) tensor
                - initial_velocity: (2,) tensor
                - field_params: dict of field parameters
        """
        if self.data is not None:
            sample = self.data[idx]
        else:
            # Generate on-the-fly
            key = random.PRNGKey(self.seed + idx)
            sample = generate_dataset_sample(
                key,
                num_wind=self.num_wind,
                num_vortex=self.num_vortex,
                num_point=self.num_point,
                config=self.config,
            )
        
        return {
            'trajectory': torch.from_numpy(np.array(sample.trajectory)).float(),
            'initial_position': torch.from_numpy(np.array(sample.initial_position)).float(),
            'initial_velocity': torch.from_numpy(np.array(sample.initial_velocity)).float(),
            'field_params': sample.field_params,
        }


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    return {
        'trajectory': torch.stack([s['trajectory'] for s in batch]),
        'initial_position': torch.stack([s['initial_position'] for s in batch]),
        'initial_velocity': torch.stack([s['initial_velocity'] for s in batch]),
        'field_params': [s['field_params'] for s in batch],
    }


def create_dataloaders(
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    batch_size: int = 32,
    num_workers: int = 0,
    **dataset_kwargs,
):
    """
    Create train/val/test dataloaders.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for TrajectoryDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = TrajectoryDataset(
        num_samples=train_size,
        seed=42,
        **dataset_kwargs,
    )
    
    val_dataset = TrajectoryDataset(
        num_samples=val_size,
        seed=12345,
        **dataset_kwargs,
    )
    
    test_dataset = TrajectoryDataset(
        num_samples=test_size,
        seed=67890,
        **dataset_kwargs,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader

