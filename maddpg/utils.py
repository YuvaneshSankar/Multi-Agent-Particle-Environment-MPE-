import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_training_results(episode_rewards, actor_losses, critic_losses, save_path=None):
    """
    Args:
        episode_rewards: List of episode rewards
        actor_losses: List of actor losses
        critic_losses: List of critic losses
        save_path: Path to save the plot (if None, displays plot)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot episode rewards
    rewards_per_episode = [np.sum(r) for r in episode_rewards]
    axes[0].plot(rewards_per_episode, alpha=0.6, label='Episode Reward')

    # Plot moving average
    window_size = min(100, len(rewards_per_episode) // 10)
    if window_size > 1:
        moving_avg = np.convolve(rewards_per_episode,
                                np.ones(window_size)/window_size,
                                mode='valid')
        axes[0].plot(range(window_size-1, len(rewards_per_episode)),
                    moving_avg,
                    'r-',
                    linewidth=2,
                    label=f'{window_size}-Episode Moving Avg')

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot actor loss
    if actor_losses:
        axes[1].plot(actor_losses, alpha=0.8)
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Actor Loss')
        axes[1].grid(True, alpha=0.3)

    # Plot critic loss
    if critic_losses:
        axes[2].plot(critic_losses, alpha=0.8)
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Critic Loss')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def save_checkpoint(agents, episode, checkpoint_dir):
    """
    Save checkpoint for all agents.

    Args:
        agents: List of MADDPG agents
        episode: Current episode number
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    for i, agent in enumerate(agents):
        filepath = os.path.join(checkpoint_dir, f'agent_{i}_episode_{episode}.pth')
        agent.save(filepath)

    print(f"Checkpoint saved at episode {episode}")


def load_checkpoint(agents, episode, checkpoint_dir):
    """
    Load checkpoint for all agents.

    Args:
        agents: List of MADDPG agents
        episode: Episode number to load
        checkpoint_dir: Directory containing checkpoints
    """
    for i, agent in enumerate(agents):
        filepath = os.path.join(checkpoint_dir, f'agent_{i}_episode_{episode}.pth')
        if os.path.exists(filepath):
            agent.load(filepath)
        else:
            print(f"Warning: Checkpoint not found for agent {i} at {filepath}")

    print(f"Checkpoint loaded from episode {episode}")


def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns.

    Args:
        rewards: List of rewards
        gamma: Discount factor

    Returns:
        returns: Discounted returns
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def set_seed(seed, env=None):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
        env: Environment (optional)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if env is not None:
        env.seed(seed)

    print(f"Random seed set to {seed}")


def get_device(use_cuda=True):
    """
    Get compute device.

    Args:
        use_cuda: Whether to use CUDA if available

    Returns:
        device: torch device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
