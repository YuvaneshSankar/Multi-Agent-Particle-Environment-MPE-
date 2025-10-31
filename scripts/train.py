import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import maddpg module
sys.path.insert(0, str(Path(__file__).parent.parent))

from maddpg.trainer import MADDPGTrainer
from maddpg.utils import set_seed, get_device, plot_training_results
from envs.mpe_env import MPEEnvWrapper


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        config: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from: {config_path}")
    return config


def setup_directories(config):
    """
    Create necessary directories for logging and checkpoints.

    Args:
        config: Configuration dictionary
    """
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    print(f"Created directories:")
    print(f"  Logs: {config['logging']['log_dir']}")
    print(f"  Checkpoints: {config['logging']['checkpoint_dir']}")


def print_config(config):
    """
    Print configuration details.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("MADDPG TRAINING CONFIGURATION")
    print("="*80)

    print("\nEnvironment:")
    print(f"  Name: {config['env']['name']}")
    print(f"  Num Agents: {config['env']['num_agents']}")
    print(f"  Max Episode Steps: {config['env']['max_episode_steps']}")

    print("\nTraining:")
    print(f"  Total Episodes: {config['training']['total_episodes']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Gamma (Discount Factor): {config['training']['gamma']}")
    print(f"  Tau (Soft Update): {config['training']['tau']}")
    print(f"  Actor Learning Rate: {config['training']['actor_lr']}")
    print(f"  Critic Learning Rate: {config['training']['critic_lr']}")
    print(f"  Noise Std Dev: {config['training']['noise_std_dev']}")
    print(f"  Update Every: {config['training']['update_every']} steps")

    print("\nReplay Buffer:")
    print(f"  Buffer Size: {config['replay_buffer']['buffer_size']}")

    print("\nLogging:")
    print(f"  Save Model Interval: {config['logging']['save_model_interval']} episodes")
    print(f"  Log Directory: {config['logging']['log_dir']}")
    print(f"  Checkpoint Directory: {config['logging']['checkpoint_dir']}")

    print("\nOther:")
    print(f"  Random Seed: {config['seed']}")
    print("="*80 + "\n")


def main():
    """Main training function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MADDPG agents on MPE environment')
    parser.add_argument('--config', type=str, default='configs/maddpg_mpe_config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config seed if provided)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save training plots at the end')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override seed if provided as argument
    if args.seed is not None:
        config['seed'] = args.seed

    # Set random seeds for reproducibility
    set_seed(config['seed'])

    # Get compute device
    if args.no_cuda:
        device = get_device(use_cuda=False)
    else:
        device = get_device(use_cuda=True)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)

    # Print configuration
    print_config(config)

    # Create directories
    setup_directories(config)

    # Initialize environment
    print("Initializing environment...")
    try:
        env = MPEEnvWrapper(
            scenario_name=config['env']['name'],
            max_episode_steps=config['env']['max_episode_steps'],
            seed=config['seed']
        )
        print(f"Environment initialized successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("\nMake sure you have installed the multiagent-particle-envs:")
        print("  pip install git+https://github.com/openai/multiagent-particle-envs.git")
        sys.exit(1)

    # Initialize trainer
    print("\nInitializing MADDPG trainer...")
    try:
        trainer = MADDPGTrainer(env, config, device)
        print("Trainer initialized successfully!")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        sys.exit(1)

    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    try:
        trainer.train(config['training']['total_episodes'])
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current progress...")
        trainer.save_models(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save training results
    print("\nSaving training results...")
    results_path = os.path.join(config['logging']['log_dir'], 'training_results.npz')
    np.savez(
        results_path,
        episode_rewards=trainer.episode_rewards,
        actor_losses=trainer.actor_losses,
        critic_losses=trainer.critic_losses
    )
    print(f"Results saved to: {results_path}")

    # Plot training results
    if args.save_plots or True:  # Always plot by default
        print("\nGenerating training plots...")
        plot_path = os.path.join(config['logging']['log_dir'], 'training_plot.png')
        plot_training_results(
            trainer.episode_rewards,
            trainer.actor_losses,
            trainer.critic_losses,
            save_path=plot_path
        )

    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Checkpoints saved in: {config['logging']['checkpoint_dir']}")
    print(f"Logs saved in: {config['logging']['log_dir']}")


if __name__ == '__main__':
    main()
