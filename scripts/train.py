#!/usr/bin/env python3
"""
TrajectoryForge Training Script

Run this script to train the inverse model on AWS or any machine.
Supports both CPU and GPU training.

Usage:
    python scripts/train.py                    # Default settings
    python scripts/train.py --full             # Full compute (GPU recommended)
    python scripts/train.py --quick            # Quick test run
    python scripts/train.py --epochs 100       # Custom epochs
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm

# TrajectoryForge imports
from model.training import Trainer, TrainingConfig
from model.curriculum import generate_sample_at_difficulty, DIFFICULTY_LEVELS
from model.augmentation import create_strong_augmentation, augment_sample
from model.architectures import create_model_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train TrajectoryForge model')
    
    # Presets
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test run (small model, few samples)')
    parser.add_argument('--full', action='store_true',
                        help='Full training (large model, lots of data)')
    
    # Custom settings
    parser.add_argument('--model', type=str, default=None,
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                        help='Model size')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of base training samples')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--augment', type=int, default=None,
                        help='Augmentation factor (samples per original)')
    
    # Output
    parser.add_argument('--output', type=str, default='models/trained_model.pkl',
                        help='Output path for saved model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def get_config(args):
    """Get training configuration based on args."""
    
    # Default: medium settings
    config = {
        'model_size': 'medium',
        'num_samples': 500,
        'augment_factor': 2,
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 3e-4,
    }
    
    # Preset: quick
    if args.quick:
        config = {
            'model_size': 'small',
            'num_samples': 100,
            'augment_factor': 1,
            'num_epochs': 20,
            'batch_size': 16,
            'learning_rate': 1e-3,
        }
    
    # Preset: full
    if args.full:
        config = {
            'model_size': 'large',
            'num_samples': 2000,
            'augment_factor': 3,
            'num_epochs': 200,
            'batch_size': 64,
            'learning_rate': 1e-4,
        }
    
    # Override with custom args
    if args.model:
        config['model_size'] = args.model
    if args.samples:
        config['num_samples'] = args.samples
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.augment:
        config['augment_factor'] = args.augment
    
    return config


def main():
    args = parse_args()
    config = get_config(args)
    
    print("=" * 60)
    print("TRAJECTORYFORGE TRAINING")
    print("=" * 60)
    
    # Hardware check
    print("\n📊 Hardware:")
    devices = jax.devices()
    print(f"   Devices: {devices}")
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("   ✅ GPU detected!")
    else:
        print("   ⚠️  CPU only (training will be slower)")
    
    # Print config
    print("\n⚙️  Configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Create model
    print(f"\n🔧 Creating {config['model_size']} model...")
    model = create_model_from_config(config['model_size'])
    
    # Count parameters
    key = random.PRNGKey(0)
    sample_input = jnp.ones((1, 100, 2))
    variables = model.init(key, sample_input, training=False)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f"   Parameters: {n_params:,}")
    
    # Generate training data
    print(f"\n📦 Generating {config['num_samples']} training samples...")
    key = random.PRNGKey(args.seed)
    train_data = []
    easy_level = DIFFICULTY_LEVELS['easy']
    
    for i in tqdm(range(config['num_samples']), desc="Generating"):
        key, subkey = random.split(key)
        sample = generate_sample_at_difficulty(subkey, easy_level)
        train_data.append(sample)
    
    # Augment data
    print(f"\n🔄 Augmenting data ({config['augment_factor']}x per sample)...")
    pipeline = create_strong_augmentation()
    augmented_data = []
    
    for sample in tqdm(train_data, desc="Augmenting"):
        augmented_data.append(sample)
        for _ in range(config['augment_factor']):
            key, subkey = random.split(key)
            aug_sample = augment_sample(subkey, sample, pipeline)
            augmented_data.append(aug_sample)
    
    print(f"   Total samples: {len(augmented_data):,}")
    
    # Create trainer
    training_config = TrainingConfig(
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        loss_type='physics',
        sim_steps=100,
        log_every=10,
    )
    
    trainer = Trainer(model, training_config, rng_seed=args.seed)
    
    # Train
    print(f"\n🚀 Training for {config['num_epochs']} epochs...")
    print(f"   Steps per epoch: {len(augmented_data) // config['batch_size']}")
    print(f"   Total steps: {config['num_epochs'] * (len(augmented_data) // config['batch_size']):,}")
    print()
    
    start_time = time.time()
    history = trainer.train(augmented_data, verbose=True)
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Time elapsed: {elapsed/60:.1f} minutes")
    print(f"   Final loss:   {history['train_loss'][-1]:.6f}")
    print(f"   Best loss:    {min(history['train_loss']):.6f}")
    print(f"   Improvement:  {history['train_loss'][0] / history['train_loss'][-1]:.1f}x")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'params': trainer.state.params,
        'config': config,
        'training_config': {
            'num_epochs': training_config.num_epochs,
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
        },
        'history': history,
        'n_params': n_params,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n💾 Model saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

