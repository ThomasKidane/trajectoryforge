"""
Curriculum Learning Module

Implements progressive training strategies that start with simple
trajectories and gradually increase complexity. This helps the model
learn basic patterns before tackling harder cases.
"""

from __future__ import annotations
from typing import NamedTuple, Callable
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField, VortexField, PointForce
from physics.simulator import simulate_positions_only


# =============================================================================
# Difficulty Levels
# =============================================================================

@dataclass
class DifficultyLevel:
    """
    Configuration for a difficulty level in curriculum learning.
    
    Controls the complexity of generated training samples.
    """
    name: str
    
    # Field strength ranges
    min_strength: float = 1.0
    max_strength: float = 5.0
    
    # Number of fields
    num_wind_fields: int = 1
    num_vortex_fields: int = 0
    num_point_forces: int = 0
    
    # Initial velocity range
    min_velocity: float = 0.5
    max_velocity: float = 1.5
    
    # Trajectory length
    num_steps: int = 100
    
    # Field size ranges
    min_field_size: float = 1.0
    max_field_size: float = 3.0
    
    # Spatial bounds for field placement
    field_bounds: tuple[float, float] = (-2.0, 4.0)
    
    # Starting position bounds
    start_bounds: tuple[float, float] = (-2.0, 0.0)


# Predefined difficulty levels
DIFFICULTY_LEVELS = {
    'trivial': DifficultyLevel(
        name='trivial',
        min_strength=1.0,
        max_strength=3.0,
        num_wind_fields=1,
        num_vortex_fields=0,
        num_point_forces=0,
        min_velocity=0.5,
        max_velocity=1.0,
        num_steps=50,
    ),
    'easy': DifficultyLevel(
        name='easy',
        min_strength=2.0,
        max_strength=5.0,
        num_wind_fields=1,
        num_vortex_fields=0,
        num_point_forces=0,
        min_velocity=0.5,
        max_velocity=1.5,
        num_steps=100,
    ),
    'medium': DifficultyLevel(
        name='medium',
        min_strength=3.0,
        max_strength=10.0,
        num_wind_fields=1,
        num_vortex_fields=1,
        num_point_forces=0,
        min_velocity=1.0,
        max_velocity=2.0,
        num_steps=150,
    ),
    'hard': DifficultyLevel(
        name='hard',
        min_strength=5.0,
        max_strength=15.0,
        num_wind_fields=2,
        num_vortex_fields=1,
        num_point_forces=1,
        min_velocity=1.0,
        max_velocity=3.0,
        num_steps=200,
    ),
    'expert': DifficultyLevel(
        name='expert',
        min_strength=5.0,
        max_strength=20.0,
        num_wind_fields=2,
        num_vortex_fields=2,
        num_point_forces=2,
        min_velocity=1.5,
        max_velocity=4.0,
        num_steps=300,
    ),
}


# =============================================================================
# Sample Generation with Difficulty
# =============================================================================

def generate_sample_at_difficulty(
    key: jax.Array,
    difficulty: DifficultyLevel,
    dt: float = 0.01,
) -> dict:
    """
    Generate a training sample at a specific difficulty level.
    
    Args:
        key: JAX random key
        difficulty: Difficulty configuration
        dt: Time step for simulation
        
    Returns:
        Training sample dict with trajectory and field parameters
    """
    keys = random.split(key, 20)  # Plenty of keys for all random ops
    key_idx = 0
    
    config = SimulationConfig(dt=dt, num_steps=difficulty.num_steps)
    
    # Generate initial conditions
    init_pos = random.uniform(
        keys[key_idx], (2,),
        minval=difficulty.start_bounds[0],
        maxval=difficulty.start_bounds[1]
    )
    key_idx += 1
    
    init_vel_dir = random.normal(keys[key_idx], (2,))
    init_vel_dir = init_vel_dir / (jnp.linalg.norm(init_vel_dir) + 1e-8)
    key_idx += 1
    
    init_vel_mag = random.uniform(
        keys[key_idx], (),
        minval=difficulty.min_velocity,
        maxval=difficulty.max_velocity
    )
    key_idx += 1
    init_vel = init_vel_dir * init_vel_mag
    
    # Generate force fields
    fields = []
    field_params = {
        'wind_centers': [],
        'wind_sizes': [],
        'wind_directions': [],
        'wind_strengths': [],
        'vortex_centers': [],
        'vortex_strengths': [],
        'vortex_radii': [],
        'point_centers': [],
        'point_strengths': [],
    }
    
    # Wind fields
    for _ in range(difficulty.num_wind_fields):
        center = random.uniform(
            keys[key_idx], (2,),
            minval=difficulty.field_bounds[0],
            maxval=difficulty.field_bounds[1]
        )
        key_idx += 1
        
        size = random.uniform(
            keys[key_idx], (2,),
            minval=difficulty.min_field_size,
            maxval=difficulty.max_field_size
        )
        key_idx += 1
        
        direction = random.normal(keys[key_idx], (2,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-8)
        key_idx += 1
        
        strength = random.uniform(
            keys[key_idx], (),
            minval=difficulty.min_strength,
            maxval=difficulty.max_strength
        )
        key_idx += 1
        
        wind = WindField(
            center=center,
            size=size,
            direction=direction,
            strength=float(strength),
            softness=0.1,
        )
        fields.append(wind)
        
        field_params['wind_centers'].append(center)
        field_params['wind_sizes'].append(size)
        field_params['wind_directions'].append(direction)
        field_params['wind_strengths'].append(strength)
    
    # Vortex fields
    for _ in range(difficulty.num_vortex_fields):
        center = random.uniform(
            keys[key_idx], (2,),
            minval=difficulty.field_bounds[0],
            maxval=difficulty.field_bounds[1]
        )
        key_idx += 1
        
        strength = random.uniform(
            keys[key_idx], (),
            minval=difficulty.min_strength,
            maxval=difficulty.max_strength
        )
        # Random sign for CW/CCW
        strength = strength * random.choice(keys[key_idx], jnp.array([-1.0, 1.0]))
        key_idx += 1
        
        radius = random.uniform(
            keys[key_idx], (),
            minval=difficulty.min_field_size,
            maxval=difficulty.max_field_size
        )
        key_idx += 1
        
        vortex = VortexField(
            center=center,
            strength=float(strength),
            radius=float(radius),
            falloff='inverse',
            softness=0.1,
        )
        fields.append(vortex)
        
        field_params['vortex_centers'].append(center)
        field_params['vortex_strengths'].append(strength)
        field_params['vortex_radii'].append(radius)
    
    # Point forces
    for _ in range(difficulty.num_point_forces):
        center = random.uniform(
            keys[key_idx], (2,),
            minval=difficulty.field_bounds[0],
            maxval=difficulty.field_bounds[1]
        )
        key_idx += 1
        
        strength = random.uniform(
            keys[key_idx], (),
            minval=difficulty.min_strength,
            maxval=difficulty.max_strength
        )
        # Random attractor/repeller
        strength = strength * random.choice(keys[key_idx], jnp.array([-1.0, 1.0]))
        key_idx += 1
        
        point = PointForce(
            center=center,
            strength=float(strength),
            falloff_power=2.0,
            max_distance=10.0,
            softness=0.1,
        )
        fields.append(point)
        
        field_params['point_centers'].append(center)
        field_params['point_strengths'].append(strength)
    
    # Simulate trajectory
    state = PhysicsState(position=init_pos, velocity=init_vel, time=0.0)
    trajectory = simulate_positions_only(state, fields, config)
    
    # Convert lists to arrays
    for k, v in field_params.items():
        if len(v) > 0:
            field_params[k] = jnp.stack(v)
        else:
            field_params[k] = None
    
    return {
        'trajectory': trajectory,
        'initial_position': init_pos,
        'initial_velocity': init_vel,
        'field_params': field_params,
        'difficulty': difficulty.name,
        'num_steps': difficulty.num_steps,
    }


def generate_curriculum_dataset(
    samples_per_level: int,
    levels: list[str] = None,
    seed: int = 42,
    dt: float = 0.01,
) -> dict[str, list[dict]]:
    """
    Generate a dataset with samples at multiple difficulty levels.
    
    Args:
        samples_per_level: Number of samples per difficulty level
        levels: List of difficulty level names (default: all levels)
        seed: Random seed
        dt: Time step for simulation
        
    Returns:
        Dict mapping difficulty names to lists of samples
    """
    if levels is None:
        levels = list(DIFFICULTY_LEVELS.keys())
    
    key = random.PRNGKey(seed)
    dataset = {}
    
    for level_name in levels:
        level = DIFFICULTY_LEVELS[level_name]
        key, subkey = random.split(key)
        keys = random.split(subkey, samples_per_level)
        
        samples = []
        for k in tqdm(keys, desc=f"Generating {level_name} samples"):
            sample = generate_sample_at_difficulty(k, level, dt)
            samples.append(sample)
        
        dataset[level_name] = samples
    
    return dataset


# =============================================================================
# Curriculum Scheduler
# =============================================================================

class CurriculumScheduler:
    """
    Manages the progression through difficulty levels during training.
    
    Supports various strategies for when to increase difficulty.
    """
    
    def __init__(
        self,
        levels: list[str] = None,
        strategy: str = 'loss_threshold',
        loss_threshold: float = 0.1,
        epochs_per_level: int = 20,
        patience: int = 5,
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            levels: Ordered list of difficulty levels
            strategy: 'loss_threshold', 'fixed_epochs', or 'patience'
            loss_threshold: Loss threshold to advance (for loss_threshold strategy)
            epochs_per_level: Fixed epochs per level (for fixed_epochs strategy)
            patience: Epochs without improvement to advance (for patience strategy)
        """
        self.levels = levels or ['trivial', 'easy', 'medium', 'hard', 'expert']
        self.strategy = strategy
        self.loss_threshold = loss_threshold
        self.epochs_per_level = epochs_per_level
        self.patience = patience
        
        self.current_level_idx = 0
        self.epochs_at_level = 0
        self.best_loss_at_level = float('inf')
        self.epochs_since_improvement = 0
        self.history = []
    
    @property
    def current_level(self) -> str:
        """Get current difficulty level name."""
        return self.levels[self.current_level_idx]
    
    @property
    def current_difficulty(self) -> DifficultyLevel:
        """Get current difficulty configuration."""
        return DIFFICULTY_LEVELS[self.current_level]
    
    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_level_idx >= len(self.levels)
    
    def step(self, loss: float) -> bool:
        """
        Update scheduler with current loss, return True if level changed.
        
        Args:
            loss: Current training loss
            
        Returns:
            True if difficulty level was increased
        """
        self.epochs_at_level += 1
        self.history.append({
            'level': self.current_level,
            'loss': loss,
            'epoch': len(self.history),
        })
        
        # Check for improvement
        if loss < self.best_loss_at_level:
            self.best_loss_at_level = loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # Check if we should advance
        should_advance = False
        
        if self.strategy == 'loss_threshold':
            should_advance = loss < self.loss_threshold
            
        elif self.strategy == 'fixed_epochs':
            should_advance = self.epochs_at_level >= self.epochs_per_level
            
        elif self.strategy == 'patience':
            should_advance = self.epochs_since_improvement >= self.patience
        
        if should_advance and self.current_level_idx < len(self.levels) - 1:
            self._advance_level()
            return True
        
        return False
    
    def _advance_level(self):
        """Advance to the next difficulty level."""
        self.current_level_idx += 1
        self.epochs_at_level = 0
        self.best_loss_at_level = float('inf')
        self.epochs_since_improvement = 0
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_level_idx = 0
        self.epochs_at_level = 0
        self.best_loss_at_level = float('inf')
        self.epochs_since_improvement = 0
        self.history = []
    
    def get_progress(self) -> dict:
        """Get curriculum progress information."""
        return {
            'current_level': self.current_level,
            'level_index': self.current_level_idx,
            'total_levels': len(self.levels),
            'epochs_at_level': self.epochs_at_level,
            'best_loss': self.best_loss_at_level,
            'progress_percent': 100 * self.current_level_idx / len(self.levels),
        }


# =============================================================================
# Curriculum Trainer
# =============================================================================

class CurriculumTrainer:
    """
    Trainer that uses curriculum learning.
    
    Automatically generates data at appropriate difficulty and
    advances through the curriculum as the model improves.
    """
    
    def __init__(
        self,
        model,
        base_trainer,
        scheduler: CurriculumScheduler,
        samples_per_level: int = 500,
        val_samples: int = 100,
        seed: int = 42,
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            model: The model to train
            base_trainer: Base Trainer instance for training
            scheduler: CurriculumScheduler for difficulty progression
            samples_per_level: Training samples to generate per level
            val_samples: Validation samples per level
            seed: Random seed for data generation
        """
        self.model = model
        self.trainer = base_trainer
        self.scheduler = scheduler
        self.samples_per_level = samples_per_level
        self.val_samples = val_samples
        self.seed = seed
        
        self.training_history = {
            'epoch': [],
            'level': [],
            'train_loss': [],
            'val_loss': [],
        }
        
        # Pre-generate data for all levels
        self._generate_all_data()
    
    def _generate_all_data(self):
        """Generate training and validation data for all levels."""
        print("Generating curriculum data...")
        self.train_data = {}
        self.val_data = {}
        
        key = random.PRNGKey(self.seed)
        
        for level_name in self.scheduler.levels:
            level = DIFFICULTY_LEVELS[level_name]
            
            # Training data
            key, subkey = random.split(key)
            keys = random.split(subkey, self.samples_per_level)
            self.train_data[level_name] = [
                generate_sample_at_difficulty(k, level)
                for k in tqdm(keys, desc=f"Train {level_name}")
            ]
            
            # Validation data
            key, subkey = random.split(key)
            keys = random.split(subkey, self.val_samples)
            self.val_data[level_name] = [
                generate_sample_at_difficulty(k, level)
                for k in tqdm(keys, desc=f"Val {level_name}")
            ]
    
    def train(
        self,
        max_epochs: int = 200,
        verbose: bool = True,
    ) -> dict:
        """
        Run curriculum training.
        
        Args:
            max_epochs: Maximum total epochs across all levels
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        epoch = 0
        
        while epoch < max_epochs and not self.scheduler.is_complete:
            level = self.scheduler.current_level
            
            if verbose:
                progress = self.scheduler.get_progress()
                print(f"\n=== Level: {level} ({progress['level_index']+1}/{progress['total_levels']}) ===")
            
            # Get data for current level
            train_data = self.train_data[level]
            val_data = self.val_data[level]
            
            # Train for one epoch
            train_loss = self.trainer.train_epoch(train_data)
            val_loss = self.trainer.evaluate(val_data)
            
            # Record history
            self.training_history['epoch'].append(epoch)
            self.training_history['level'].append(level)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Update scheduler
            level_changed = self.scheduler.step(val_loss)
            
            if verbose:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                if level_changed:
                    print(f">>> Advancing to level: {self.scheduler.current_level}")
            
            epoch += 1
        
        if verbose:
            if self.scheduler.is_complete:
                print("\n✅ Curriculum complete!")
            else:
                print(f"\n⏹ Training stopped at epoch {epoch}")
        
        return self.training_history


# =============================================================================
# Mixed Difficulty Sampling
# =============================================================================

def sample_mixed_difficulty_batch(
    key: jax.Array,
    datasets: dict[str, list[dict]],
    batch_size: int,
    level_weights: dict[str, float] = None,
) -> list[dict]:
    """
    Sample a batch with mixed difficulty levels.
    
    This can be used for training on diverse data or for
    maintaining performance on easier levels while learning harder ones.
    
    Args:
        key: JAX random key
        datasets: Dict mapping level names to sample lists
        batch_size: Total batch size
        level_weights: Optional weights for each level (default: uniform)
        
    Returns:
        List of samples
    """
    levels = list(datasets.keys())
    
    if level_weights is None:
        weights = jnp.ones(len(levels)) / len(levels)
    else:
        weights = jnp.array([level_weights.get(l, 1.0) for l in levels])
        weights = weights / weights.sum()
    
    # Sample level for each item in batch
    key, subkey = random.split(key)
    level_indices = random.choice(
        subkey, len(levels), shape=(batch_size,), p=weights
    )
    
    # Sample from each level
    batch = []
    for i, level_idx in enumerate(level_indices):
        level = levels[int(level_idx)]
        key, subkey = random.split(key)
        sample_idx = random.randint(subkey, (), 0, len(datasets[level]))
        batch.append(datasets[level][int(sample_idx)])
    
    return batch


def create_progressive_weights(
    current_level_idx: int,
    num_levels: int,
    focus_current: float = 0.5,
    include_previous: bool = True,
) -> dict[str, float]:
    """
    Create level weights that focus on current level but include easier levels.
    
    This helps prevent catastrophic forgetting of easier cases.
    
    Args:
        current_level_idx: Index of current difficulty level
        num_levels: Total number of levels
        focus_current: Weight for current level (0-1)
        include_previous: Whether to include previous levels
        
    Returns:
        Dict of level weights
    """
    levels = list(DIFFICULTY_LEVELS.keys())[:num_levels]
    weights = {}
    
    if include_previous:
        # Distribute remaining weight among previous levels
        remaining = 1.0 - focus_current
        for i, level in enumerate(levels):
            if i < current_level_idx:
                # Previous levels get decreasing weight
                weights[level] = remaining * (0.5 ** (current_level_idx - i))
            elif i == current_level_idx:
                weights[level] = focus_current
            else:
                weights[level] = 0.0
    else:
        # Only current level
        for i, level in enumerate(levels):
            weights[level] = focus_current if i == current_level_idx else 0.0
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights

