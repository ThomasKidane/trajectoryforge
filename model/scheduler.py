"""
Learning Rate Scheduling Module

Provides various learning rate schedules for training optimization.
Includes warmup, decay, and curriculum-aware schedules.
"""

from __future__ import annotations
from typing import Callable
import jax.numpy as jnp
import optax


# =============================================================================
# Basic Schedules
# =============================================================================

def constant_schedule(learning_rate: float) -> optax.Schedule:
    """Constant learning rate (no scheduling)."""
    return optax.constant_schedule(learning_rate)


def linear_decay_schedule(
    init_lr: float,
    end_lr: float,
    total_steps: int,
) -> optax.Schedule:
    """
    Linear decay from init_lr to end_lr over total_steps.
    
    Args:
        init_lr: Initial learning rate
        end_lr: Final learning rate
        total_steps: Number of steps for decay
        
    Returns:
        Optax schedule
    """
    return optax.linear_schedule(
        init_value=init_lr,
        end_value=end_lr,
        transition_steps=total_steps,
    )


def exponential_decay_schedule(
    init_lr: float,
    decay_rate: float,
    decay_steps: int,
    staircase: bool = False,
) -> optax.Schedule:
    """
    Exponential decay: lr = init_lr * decay_rate^(step / decay_steps)
    
    Args:
        init_lr: Initial learning rate
        decay_rate: Multiplicative decay factor
        decay_steps: Steps between decay applications
        staircase: If True, decay in discrete steps
        
    Returns:
        Optax schedule
    """
    return optax.exponential_decay(
        init_value=init_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )


def cosine_decay_schedule(
    init_lr: float,
    total_steps: int,
    alpha: float = 0.0,
) -> optax.Schedule:
    """
    Cosine annealing schedule.
    
    lr = alpha + (init_lr - alpha) * 0.5 * (1 + cos(pi * step / total_steps))
    
    Args:
        init_lr: Initial learning rate
        total_steps: Total training steps
        alpha: Minimum learning rate (default 0)
        
    Returns:
        Optax schedule
    """
    return optax.cosine_decay_schedule(
        init_value=init_lr,
        decay_steps=total_steps,
        alpha=alpha,
    )


# =============================================================================
# Warmup Schedules
# =============================================================================

def warmup_constant_schedule(
    peak_lr: float,
    warmup_steps: int,
) -> optax.Schedule:
    """
    Linear warmup to peak_lr, then constant.
    
    Args:
        peak_lr: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        
    Returns:
        Optax schedule
    """
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, peak_lr, warmup_steps),
            optax.constant_schedule(peak_lr),
        ],
        boundaries=[warmup_steps],
    )


def warmup_linear_decay_schedule(
    peak_lr: float,
    end_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """
    Linear warmup, then linear decay.
    
    Args:
        peak_lr: Peak learning rate after warmup
        end_lr: Final learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        
    Returns:
        Optax schedule
    """
    decay_steps = total_steps - warmup_steps
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, peak_lr, warmup_steps),
            optax.linear_schedule(peak_lr, end_lr, decay_steps),
        ],
        boundaries=[warmup_steps],
    )


def warmup_cosine_decay_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """
    Linear warmup, then cosine decay.
    
    This is a popular schedule for transformer training.
    
    Args:
        peak_lr: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        end_lr: Final learning rate (default 0)
        
    Returns:
        Optax schedule
    """
    decay_steps = total_steps - warmup_steps
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, peak_lr, warmup_steps),
            optax.cosine_decay_schedule(peak_lr, decay_steps, alpha=end_lr),
        ],
        boundaries=[warmup_steps],
    )


def warmup_exponential_decay_schedule(
    peak_lr: float,
    warmup_steps: int,
    decay_rate: float,
    decay_steps: int,
) -> optax.Schedule:
    """
    Linear warmup, then exponential decay.
    
    Args:
        peak_lr: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        decay_rate: Exponential decay rate
        decay_steps: Steps between decay applications
        
    Returns:
        Optax schedule
    """
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, peak_lr, warmup_steps),
            optax.exponential_decay(peak_lr, decay_steps, decay_rate),
        ],
        boundaries=[warmup_steps],
    )


# =============================================================================
# Advanced Schedules
# =============================================================================

def cyclic_schedule(
    min_lr: float,
    max_lr: float,
    cycle_steps: int,
    mode: str = 'triangular',
) -> Callable[[int], float]:
    """
    Cyclic learning rate schedule.
    
    Args:
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        cycle_steps: Steps per cycle
        mode: 'triangular', 'triangular2', or 'exp_range'
        
    Returns:
        Schedule function
    """
    def schedule(step: int) -> float:
        cycle = jnp.floor(step / cycle_steps)
        x = jnp.abs(step / cycle_steps - 2 * cycle + 1)
        
        if mode == 'triangular':
            scale = 1.0
        elif mode == 'triangular2':
            scale = 1.0 / (2.0 ** cycle)
        elif mode == 'exp_range':
            scale = 0.99994 ** step
        else:
            scale = 1.0
        
        return min_lr + (max_lr - min_lr) * jnp.maximum(0, (1 - x)) * scale
    
    return schedule


def one_cycle_schedule(
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Callable[[int], float]:
    """
    One-cycle learning rate schedule (Smith & Topin, 2019).
    
    Increases LR from max_lr/div_factor to max_lr, then decreases to max_lr/final_div_factor.
    
    Args:
        max_lr: Maximum learning rate
        total_steps: Total training steps
        pct_start: Percentage of cycle spent increasing LR
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = max_lr / final_div_factor
        
    Returns:
        Schedule function
    """
    init_lr = max_lr / div_factor
    final_lr = max_lr / final_div_factor
    up_steps = int(total_steps * pct_start)
    down_steps = total_steps - up_steps
    
    def schedule(step: int) -> float:
        step = jnp.minimum(step, total_steps - 1)
        
        # Increasing phase
        increasing = step < up_steps
        up_pct = step / up_steps
        up_lr = init_lr + (max_lr - init_lr) * up_pct
        
        # Decreasing phase
        down_pct = (step - up_steps) / down_steps
        down_lr = max_lr - (max_lr - final_lr) * down_pct
        
        return jnp.where(increasing, up_lr, down_lr)
    
    return schedule


def reduce_on_plateau_schedule(
    init_lr: float,
    factor: float = 0.5,
    patience: int = 10,
    min_lr: float = 1e-6,
) -> dict:
    """
    Returns configuration for reduce-on-plateau scheduling.
    
    Note: This requires manual integration in training loop
    since it depends on validation loss.
    
    Args:
        init_lr: Initial learning rate
        factor: Multiplicative factor for reduction
        patience: Number of epochs to wait before reducing
        min_lr: Minimum learning rate
        
    Returns:
        Configuration dict for manual implementation
    """
    return {
        'type': 'reduce_on_plateau',
        'init_lr': init_lr,
        'factor': factor,
        'patience': patience,
        'min_lr': min_lr,
    }


# =============================================================================
# Curriculum-Aware Schedules
# =============================================================================

def curriculum_reset_schedule(
    base_schedule_fn: Callable,
    level_steps: list[int],
    warmup_steps_per_level: int = 100,
) -> Callable[[int], float]:
    """
    Schedule that resets warmup when curriculum level changes.
    
    Args:
        base_schedule_fn: Function that creates a schedule given total steps
        level_steps: List of steps where each level starts
        warmup_steps_per_level: Warmup steps at start of each level
        
    Returns:
        Schedule function
    """
    def schedule(step: int) -> float:
        # Find current level
        level = 0
        level_start = 0
        for i, boundary in enumerate(level_steps):
            if step >= boundary:
                level = i + 1
                level_start = boundary
        
        # Steps within current level
        steps_in_level = step - level_start
        
        # Determine total steps for this level
        if level < len(level_steps):
            level_total = level_steps[level] - level_start
        else:
            level_total = 10000  # Default for final level
        
        # Create schedule for this level
        level_schedule = base_schedule_fn(level_total)
        
        return level_schedule(steps_in_level)
    
    return schedule


def progressive_lr_schedule(
    base_lr: float,
    num_levels: int,
    lr_multiplier: float = 0.5,
    warmup_steps: int = 100,
) -> Callable[[int, int], float]:
    """
    Learning rate that decreases as curriculum progresses.
    
    Harder levels get lower learning rates for finer adjustments.
    
    Args:
        base_lr: Base learning rate for first level
        num_levels: Total number of curriculum levels
        lr_multiplier: Multiplier applied at each level
        warmup_steps: Warmup steps at start of each level
        
    Returns:
        Schedule function taking (step, level) as arguments
    """
    def schedule(step: int, level: int) -> float:
        # Compute LR for this level
        level_lr = base_lr * (lr_multiplier ** level)
        
        # Apply warmup within level
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        
        return level_lr * warmup_factor
    
    return schedule


# =============================================================================
# Optimizer Factory
# =============================================================================

def create_optimizer(
    schedule: optax.Schedule | Callable,
    optimizer_type: str = 'adam',
    weight_decay: float = 0.0,
    gradient_clip: float = None,
    **kwargs,
) -> optax.GradientTransformation:
    """
    Create an optimizer with the given schedule and configuration.
    
    Args:
        schedule: Learning rate schedule
        optimizer_type: 'adam', 'adamw', 'sgd', 'rmsprop'
        weight_decay: Weight decay coefficient
        gradient_clip: Max gradient norm (None for no clipping)
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optax optimizer
    """
    transforms = []
    
    # Gradient clipping
    if gradient_clip is not None:
        transforms.append(optax.clip_by_global_norm(gradient_clip))
    
    # Main optimizer
    if optimizer_type == 'adam':
        transforms.append(optax.scale_by_adam(**kwargs))
    elif optimizer_type == 'adamw':
        transforms.append(optax.scale_by_adam(**kwargs))
        if weight_decay > 0:
            transforms.append(optax.add_decayed_weights(weight_decay))
    elif optimizer_type == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        if momentum > 0:
            transforms.append(optax.trace(decay=momentum))
    elif optimizer_type == 'rmsprop':
        transforms.append(optax.scale_by_rms(**kwargs))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Learning rate schedule
    transforms.append(optax.scale_by_schedule(schedule))
    transforms.append(optax.scale(-1.0))  # Gradient descent
    
    return optax.chain(*transforms)


def create_adam_with_warmup(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create Adam optimizer with warmup and cosine decay.
    
    This is a common configuration for training neural networks.
    
    Args:
        peak_lr: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        weight_decay: Weight decay coefficient
        gradient_clip: Max gradient norm
        
    Returns:
        Optax optimizer
    """
    schedule = warmup_cosine_decay_schedule(peak_lr, warmup_steps, total_steps)
    
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )


# =============================================================================
# Schedule Visualization (for debugging)
# =============================================================================

def visualize_schedule(
    schedule: Callable[[int], float],
    total_steps: int,
    title: str = "Learning Rate Schedule",
) -> tuple:
    """
    Generate data for visualizing a learning rate schedule.
    
    Args:
        schedule: Schedule function
        total_steps: Number of steps to visualize
        title: Plot title
        
    Returns:
        Tuple of (steps, learning_rates, title)
    """
    steps = list(range(total_steps))
    lrs = [float(schedule(s)) for s in steps]
    return steps, lrs, title

