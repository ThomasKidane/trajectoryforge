"""
Training Module

End-to-end training for the inverse model using differentiable physics.
Supports both supervised learning and physics-based loss.
"""

from __future__ import annotations
from typing import NamedTuple, Callable, Any
from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from tqdm import tqdm

from physics.state import PhysicsState, SimulationConfig
from physics.fields import WindField, VortexField, PointForce
from physics.simulator import simulate_positions_only, trajectory_loss_same_length


# =============================================================================
# Training State
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended training state with additional metrics."""
    batch_stats: Any = None


def create_train_state(
    model: nn.Module,
    rng: jax.Array,
    learning_rate: float = 1e-3,
    sample_input_shape: tuple = (100, 2),
) -> TrainState:
    """
    Initialize training state.
    
    Args:
        model: Flax model
        rng: Random key for initialization
        learning_rate: Learning rate for optimizer
        sample_input_shape: Shape of sample trajectory input
        
    Returns:
        Initialized TrainState
    """
    # Initialize model
    sample_input = jnp.ones((1,) + sample_input_shape)
    variables = model.init(rng, sample_input, training=False)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables.get('batch_stats'),
    )


# =============================================================================
# Loss Functions for Training
# =============================================================================

def supervised_loss(
    params: dict,
    apply_fn: Callable,
    batch: dict,
    training: bool = True,
) -> tuple[float, dict]:
    """
    Supervised loss: predict field parameters from trajectory.
    
    This is a simple MSE loss on the predicted parameters.
    Requires ground truth field parameters in the batch.
    """
    trajectories = batch['trajectory']
    target_params = batch['field_params']
    
    # Forward pass
    predictions = apply_fn({'params': params}, trajectories, training=training)
    
    # MSE loss on parameters
    loss = 0.0
    
    # Wind parameters
    if 'wind_strength' in target_params:
        loss += jnp.mean((predictions.wind_strengths - target_params['wind_strength']) ** 2)
        loss += jnp.mean((predictions.wind_centers - target_params['wind_center']) ** 2)
    
    metrics = {'loss': loss}
    return loss, metrics


def physics_loss(
    params: dict,
    apply_fn: Callable,
    batch: dict,
    initial_state: PhysicsState,
    config: SimulationConfig,
    training: bool = True,
) -> tuple[float, dict]:
    """
    Physics-based loss: simulate predicted fields and compare trajectories.
    
    This loss backpropagates through the differentiable physics simulation,
    allowing the model to learn from trajectory matching directly.
    """
    target_trajectories = batch['trajectory']
    initial_positions = batch['initial_position']
    initial_velocities = batch['initial_velocity']
    
    batch_size = target_trajectories.shape[0]
    
    # Forward pass - get field predictions
    predictions = apply_fn({'params': params}, target_trajectories, training=training)
    
    # Simulate each predicted configuration and compute loss
    total_loss = 0.0
    
    for i in range(batch_size):
        # Create wind field from prediction
        wind = WindField(
            center=predictions.wind_centers[i, 0],  # First wind field
            size=predictions.wind_sizes[i, 0],
            direction=predictions.wind_directions[i, 0],
            strength=predictions.wind_strengths[i, 0],
            softness=0.1,
        )
        
        # Create initial state
        state = PhysicsState(
            position=initial_positions[i],
            velocity=initial_velocities[i],
            time=0.0,
        )
        
        # Simulate
        predicted_traj = simulate_positions_only(state, [wind], config)
        
        # Trajectory loss
        loss = trajectory_loss_same_length(predicted_traj, target_trajectories[i])
        total_loss += loss
    
    total_loss = total_loss / batch_size
    
    metrics = {'loss': total_loss}
    return total_loss, metrics


def physics_loss_vmapped(
    params: dict,
    apply_fn: Callable,
    batch: dict,
    config: SimulationConfig,
    training: bool = True,
    rng: jax.Array = None,
) -> tuple[float, dict]:
    """
    Vectorized physics loss for better performance.
    
    Uses vmap to parallelize simulation across batch.
    """
    target_trajectories = batch['trajectory']
    initial_positions = batch['initial_position']
    initial_velocities = batch['initial_velocity']
    
    # Forward pass (with dropout RNG if training)
    if training and rng is not None:
        predictions = apply_fn(
            {'params': params}, 
            target_trajectories, 
            training=training,
            rngs={'dropout': rng}
        )
    else:
        predictions = apply_fn({'params': params}, target_trajectories, training=False)
    
    # Define single-sample simulation
    def simulate_single(init_pos, init_vel, wind_center, wind_size, wind_dir, wind_strength, target):
        wind = WindField(
            center=wind_center,
            size=wind_size,
            direction=wind_dir,
            strength=wind_strength,
            softness=0.1,
        )
        state = PhysicsState(position=init_pos, velocity=init_vel, time=0.0)
        predicted = simulate_positions_only(state, [wind], config)
        return trajectory_loss_same_length(predicted, target)
    
    # Vectorize over batch
    losses = vmap(simulate_single)(
        initial_positions,
        initial_velocities,
        predictions.wind_centers[:, 0],
        predictions.wind_sizes[:, 0],
        predictions.wind_directions[:, 0],
        predictions.wind_strengths[:, 0],
        target_trajectories,
    )
    
    total_loss = jnp.mean(losses)
    metrics = {'loss': total_loss, 'per_sample_loss': losses}
    return total_loss, metrics


# =============================================================================
# Training Step
# =============================================================================

@partial(jit, static_argnums=(2, 3))
def train_step_supervised(
    state: TrainState,
    batch: dict,
    apply_fn: Callable,
    training: bool = True,
) -> tuple[TrainState, dict]:
    """Single supervised training step."""
    
    def loss_fn(params):
        return supervised_loss(params, apply_fn, batch, training)
    
    (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


def train_step_physics(
    state: TrainState,
    batch: dict,
    config: SimulationConfig,
    rng: jax.Array,
) -> tuple[TrainState, dict, jax.Array]:
    """Single physics-based training step."""
    rng, dropout_rng = random.split(rng)
    
    def loss_fn(params):
        return physics_loss_vmapped(
            params, state.apply_fn, batch, config, training=True, rng=dropout_rng
        )
    
    (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics, rng


# =============================================================================
# Training Loop
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    loss_type: str = 'physics'  # 'supervised' or 'physics'
    log_every: int = 10
    eval_every: int = 20
    checkpoint_every: int = 50
    
    # Physics simulation config
    sim_dt: float = 0.01
    sim_steps: int = 200


class Trainer:
    """
    Trainer for the inverse model.
    
    Supports both supervised and physics-based training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        rng_seed: int = 42,
    ):
        self.model = model
        self.config = config
        self.rng = random.PRNGKey(rng_seed)
        
        self.sim_config = SimulationConfig(
            dt=config.sim_dt,
            num_steps=config.sim_steps,
        )
        
        # Initialize training state
        self.rng, init_rng = random.split(self.rng)
        self.state = create_train_state(
            model,
            init_rng,
            learning_rate=config.learning_rate,
            sample_input_shape=(config.sim_steps, 2),
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': [],
        }
    
    def train_epoch(self, train_data: list[dict]) -> float:
        """Train for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        self.rng, shuffle_rng = random.split(self.rng)
        indices = random.permutation(shuffle_rng, len(train_data))
        
        # Process batches
        for i in range(0, len(train_data), self.config.batch_size):
            batch_indices = indices[i:i + self.config.batch_size]
            batch = self._collate_batch([train_data[j] for j in batch_indices])
            
            if self.config.loss_type == 'physics':
                self.rng, step_rng = random.split(self.rng)
                self.state, metrics, self.rng = train_step_physics(
                    self.state, batch, self.sim_config, step_rng
                )
            else:
                self.state, metrics = train_step_supervised(
                    self.state, batch, self.state.apply_fn
                )
            
            epoch_loss += float(metrics['loss'])
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def evaluate(self, val_data: list[dict]) -> float:
        """Evaluate on validation data."""
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(val_data), self.config.batch_size):
            batch = self._collate_batch(val_data[i:i + self.config.batch_size])
            
            if self.config.loss_type == 'physics':
                loss, _ = physics_loss_vmapped(
                    self.state.params, self.state.apply_fn, batch,
                    self.sim_config, training=False
                )
            else:
                loss, _ = supervised_loss(
                    self.state.params, self.state.apply_fn, batch, training=False
                )
            
            total_loss += float(loss)
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_data: list[dict],
        val_data: list[dict] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            train_data: List of training samples
            val_data: Optional validation data
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        iterator = range(self.config.num_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")
        
        for epoch in iterator:
            # Train
            train_loss = self.train_epoch(train_data)
            self.history['train_loss'].append(train_loss)
            self.history['epoch'].append(epoch)
            
            # Evaluate
            if val_data is not None and epoch % self.config.eval_every == 0:
                val_loss = self.evaluate(val_data)
                self.history['val_loss'].append(val_loss)
            
            # Log
            if verbose and epoch % self.config.log_every == 0:
                msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
                if val_data is not None and len(self.history['val_loss']) > 0:
                    msg += f", val_loss={self.history['val_loss'][-1]:.4f}"
                iterator.set_postfix_str(msg)
        
        return self.history
    
    def _collate_batch(self, samples: list[dict]) -> dict:
        """Collate samples into a batch."""
        batch = {
            'trajectory': jnp.stack([s['trajectory'] for s in samples]),
            'initial_position': jnp.stack([s['initial_position'] for s in samples]),
            'initial_velocity': jnp.stack([s['initial_velocity'] for s in samples]),
        }
        
        # Handle field_params - filter out None values
        if 'field_params' in samples[0] and samples[0]['field_params']:
            field_params = {}
            for k in samples[0]['field_params'].keys():
                values = [s['field_params'].get(k) for s in samples]
                # Only stack if all values are not None
                if all(v is not None for v in values):
                    field_params[k] = jnp.stack(values)
            batch['field_params'] = field_params
        else:
            batch['field_params'] = {}
        
        return batch
    
    def predict(self, trajectory: jnp.ndarray):
        """Make a prediction for a single trajectory."""
        if trajectory.ndim == 2:
            trajectory = trajectory[None, ...]
        
        return self.state.apply_fn(
            {'params': self.state.params},
            trajectory,
            training=False,
        )


# =============================================================================
# Data Generation for Training
# =============================================================================

def generate_training_sample(
    key: jax.Array,
    config: SimulationConfig,
    field_type: str = 'wind',
) -> dict:
    """
    Generate a single training sample.
    
    Creates random field configuration, simulates trajectory,
    returns both for training.
    """
    keys = random.split(key, 5)
    
    # Random initial conditions
    init_pos = random.uniform(keys[0], (2,), minval=-2.0, maxval=0.0)
    init_vel_dir = random.normal(keys[1], (2,))
    init_vel_dir = init_vel_dir / (jnp.linalg.norm(init_vel_dir) + 1e-8)
    init_vel_mag = random.uniform(keys[2], (), minval=1.0, maxval=2.0)
    init_vel = init_vel_dir * init_vel_mag
    
    # Random wind field
    wind_center = random.uniform(keys[3], (2,), minval=-2.0, maxval=3.0)
    wind_size = random.uniform(keys[3], (2,), minval=1.0, maxval=4.0)
    wind_dir = random.normal(keys[4], (2,))
    wind_dir = wind_dir / (jnp.linalg.norm(wind_dir) + 1e-8)
    wind_strength = random.uniform(keys[4], (), minval=2.0, maxval=15.0)
    
    # Create field and simulate
    wind = WindField(
        center=wind_center,
        size=wind_size,
        direction=wind_dir,
        strength=float(wind_strength),
        softness=0.1,
    )
    
    state = PhysicsState(position=init_pos, velocity=init_vel, time=0.0)
    trajectory = simulate_positions_only(state, [wind], config)
    
    return {
        'trajectory': trajectory,
        'initial_position': init_pos,
        'initial_velocity': init_vel,
        'field_params': {
            'wind_center': wind_center,
            'wind_size': wind_size,
            'wind_direction': wind_dir,
            'wind_strength': wind_strength,
        },
    }


def generate_training_dataset(
    num_samples: int,
    config: SimulationConfig,
    seed: int = 42,
    field_type: str = 'wind',
) -> list[dict]:
    """Generate a dataset of training samples."""
    key = random.PRNGKey(seed)
    keys = random.split(key, num_samples)
    
    samples = []
    for k in tqdm(keys, desc="Generating data"):
        sample = generate_training_sample(k, config, field_type)
        samples.append(sample)
    
    return samples


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_model(
    model_state: TrainState,
    test_samples: list[dict],
    config: SimulationConfig,
) -> dict:
    """
    Evaluate model on test samples.
    
    Returns metrics including trajectory error and parameter error.
    """
    trajectory_errors = []
    
    for sample in test_samples:
        # Get prediction
        trajectory = sample['trajectory'][None, ...]
        prediction = model_state.apply_fn(
            {'params': model_state.params},
            trajectory,
            training=False,
        )
        
        # Simulate predicted field
        wind = WindField(
            center=prediction.wind_centers[0, 0],
            size=prediction.wind_sizes[0, 0],
            direction=prediction.wind_directions[0, 0],
            strength=float(prediction.wind_strengths[0, 0]),
            softness=0.1,
        )
        
        state = PhysicsState(
            position=sample['initial_position'],
            velocity=sample['initial_velocity'],
            time=0.0,
        )
        
        predicted_traj = simulate_positions_only(state, [wind], config)
        
        # Compute error
        error = trajectory_loss_same_length(predicted_traj, sample['trajectory'])
        trajectory_errors.append(float(error))
    
    return {
        'mean_trajectory_error': np.mean(trajectory_errors),
        'std_trajectory_error': np.std(trajectory_errors),
        'median_trajectory_error': np.median(trajectory_errors),
        'trajectory_errors': trajectory_errors,
    }

