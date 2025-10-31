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
from maddpg.utils import set_seed, get_device
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


def print_test_info(config, checkpoint_episode):
    """
    Print test information.

    Args:
        config: Configuration dictionary
        checkpoint_episode: Episode number of checkpoint being loaded
    """
    print("\n" + "="*80)
    print("MADDPG TESTING CONFIGURATION")
    print("="*80)

    print("\nEnvironment:")
    print(f"  Name: {config['env']['name']}")
    print(f"  Num Agents: {config['env']['num_agents']}")
    print(f"  Max Episode Steps: {config['env']['max_episode_steps']}")

    print("\nCheckpoint:")
    print(f"  Episode: {checkpoint_episode}")
    print(f"  Directory: {config['logging']['checkpoint_dir']}")

    print("="*80 + "\n")


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        episode_num: Episode number of latest checkpoint, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    if not checkpoint_files:
        return None

    # Extract episode numbers from filenames
    episode_numbers = []
    for f in checkpoint_files:
        try:
            # Extract episode number from filename like 'agent_0_episode_100.pth'
            episode_num = int(f.split('episode_')[-1].replace('.pth', ''))
            episode_numbers.append(episode_num)
        except ValueError:
            continue

    if episode_numbers:
        return max(episode_numbers)

    return None


def main():
    """Main testing function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test trained MADDPG agents on MPE environment')
    parser.add_argument('--config', type=str, default='configs/maddpg_mpe_config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--checkpoint', type=int, default=None,
                       help='Episode number of checkpoint to load (uses latest if not specified)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during testing')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for testing')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Use provided seed or config seed
    seed = args.seed if args.seed is not None else config['seed']
    set_seed(seed)

    # Get compute device
    if args.no_cuda:
        device = get_device(use_cuda=False)
    else:
        device = get_device(use_cuda=True)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)

    # Find checkpoint to load
    checkpoint_dir = config['logging']['checkpoint_dir']

    if args.checkpoint is not None:
        checkpoint_episode = args.checkpoint
    else:
        checkpoint_episode = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_episode is None:
            print(f"Error: No checkpoints found in {checkpoint_dir}")
            print("Please train the models first using: python scripts/train.py")
            sys.exit(1)

    print_test_info(config, checkpoint_episode)

    # Initialize environment
    print("Initializing environment...")
    try:
        env = MPEEnvWrapper(
            scenario_name=config['env']['name'],
            max_episode_steps=config['env']['max_episode_steps'],
            seed=seed
        )
        print(f"Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("\nMake sure you have installed the multiagent-particle-envs:")
        print("  pip install git+https://github.com/openai/multiagent-particle-envs.git")
        sys.exit(1)

    # Initialize trainer (needed for evaluation)
    print("Initializing trainer...")
    try:
        trainer = MADDPGTrainer(env, config, device)
        print("Trainer initialized successfully!")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        sys.exit(1)

    # Load trained models
    print(f"\nLoading models from episode {checkpoint_episode}...")
    try:
        trainer.load_models(checkpoint_episode)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print(f"Make sure checkpoint files exist in {checkpoint_dir}")
        sys.exit(1)

    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80 + "\n")

    try:
        avg_reward = trainer.evaluate(
            n_episodes=args.episodes,
            render=args.render
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nAverage reward over {args.episodes} episodes: {avg_reward:.2f}")


if __name__ == '__main__':
    main()
