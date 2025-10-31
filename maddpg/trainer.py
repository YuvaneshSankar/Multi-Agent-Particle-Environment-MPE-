import torch
import numpy as np
import os
from maddpg.agent import MADDPGAgent
from maddpg.replay_buffer import ReplayBuffer
from maddpg.noise import OUNoise


class MADDPGTrainer:

    def __init__(self, env, config, device):
        """
        Initialize MADDPG trainer.

        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: Device to run computations on
        """
        self.env = env
        self.config = config
        self.device = device

        # Environment properties
        self.n_agents = config['env']['num_agents']
        self.max_episode_steps = config['env']['max_episode_steps']

        # Get observation and action dimensions from environment
        self.obs_dims = [self.env.observation_space[i].shape[0] for i in range(self.n_agents)]
        self.action_dims = [self.env.action_space[i].shape[0] for i in range(self.n_agents)]

        # Training parameters
        self.gamma = config['training']['gamma']
        self.tau = config['training']['tau']
        self.batch_size = config['training']['batch_size']
        self.update_every = config['training']['update_every']
        self.noise_std = config['training']['noise_std_dev']

        # Create agents
        self.agents = []
        for i in range(self.n_agents):
            agent = MADDPGAgent(
                agent_id=i,
                obs_dim=self.obs_dims[i],
                action_dim=self.action_dims[i],
                n_agents=self.n_agents,
                actor_lr=config['training']['actor_lr'],
                critic_lr=config['training']['critic_lr'],
                tau=self.tau,
                gamma=self.gamma,
                device=device
            )
            self.agents.append(agent)

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=config['replay_buffer']['buffer_size'],
            n_agents=self.n_agents
        )

        # Create exploration noise for each agent
        self.noises = [OUNoise(self.action_dims[i], sigma=self.noise_std)
                       for i in range(self.n_agents)]

        # Logging
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []

    def reset_noise(self):
        """Reset exploration noise for all agents."""
        for noise in self.noises:
            noise.reset()

    def get_actions(self, observations, add_noise=True, noise_scale=1.0):
        """
        Get actions from all agents.

        Args:
            observations: List of observations for each agent
            add_noise: Whether to add exploration noise
            noise_scale: Scale factor for noise (decays over training)

        Returns:
            actions: List of actions for each agent
        """
        actions = []
        for i, agent in enumerate(self.agents):
            if add_noise:
                noise = self.noises[i].sample() * noise_scale
            else:
                noise = 0.0
            action = agent.act(observations[i], noise=noise)
            actions.append(action)
        return actions

    def update_agents(self, step):
        """
        Update all agents using samples from replay buffer.

        Args:
            step: Current training step

        Returns:
            actor_loss: Average actor loss across agents
            critic_loss: Average critic loss across agents
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None, None

        # Sample batch from replay buffer
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        obs_tensors = [torch.FloatTensor(obs_batch[i]).to(self.device)
                       for i in range(self.n_agents)]
        actions_tensors = [torch.FloatTensor(actions_batch[i]).to(self.device)
                          for i in range(self.n_agents)]
        rewards_tensors = [torch.FloatTensor(rewards_batch[i]).to(self.device)
                          for i in range(self.n_agents)]
        next_obs_tensors = [torch.FloatTensor(next_obs_batch[i]).to(self.device)
                           for i in range(self.n_agents)]
        dones_tensors = [torch.FloatTensor(dones_batch[i]).to(self.device)
                        for i in range(self.n_agents)]

        # Concatenate all observations and actions for centralized critic
        obs_full = torch.cat(obs_tensors, dim=1)
        actions_full = torch.cat(actions_tensors, dim=1)
        next_obs_full = torch.cat(next_obs_tensors, dim=1)

        # Get next actions from target actors for computing target Q-values
        with torch.no_grad():
            next_actions = [agent.target_act(next_obs_tensors[i])
                           for i, agent in enumerate(self.agents)]
            next_actions_full = torch.cat(next_actions, dim=1)

        total_actor_loss = 0
        total_critic_loss = 0

        # Update each agent
        for i, agent in enumerate(self.agents):
            # Update critic
            critic_loss = agent.update_critic(
                obs_full,
                actions_full,
                rewards_tensors[i],
                next_obs_full,
                next_actions_full,
                dones_tensors[i]
            )
            total_critic_loss += critic_loss

            # Update actor
            # Get actions from current policy for all agents
            current_actions = []
            for j, other_agent in enumerate(self.agents):
                if j == i:
                    # For this agent, use current policy and keep gradients
                    current_actions.append(other_agent.actor(obs_tensors[j]))
                else:
                    # For other agents, detach gradients
                    current_actions.append(other_agent.actor(obs_tensors[j]).detach())

            actor_loss = agent.update_actor(obs_full, current_actions, i)
            total_actor_loss += actor_loss

            # Soft update target networks
            agent.update_targets()

        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents

        return avg_actor_loss, avg_critic_loss

    def train(self, total_episodes):
        """
        Main training loop.

        Args:
            total_episodes: Number of episodes to train for
        """
        print(f"Starting MADDPG training for {total_episodes} episodes...")
        print(f"Environment: {self.config['env']['name']}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Device: {self.device}")

        step = 0

        for episode in range(total_episodes):
            # Reset environment and noise
            obs = self.env.reset()
            self.reset_noise()

            episode_reward = np.zeros(self.n_agents)
            episode_actor_loss = []
            episode_critic_loss = []

            # Decay noise over time
            noise_scale = max(0.1, 1.0 - episode / (total_episodes * 0.5))

            for t in range(self.max_episode_steps):
                # Get actions from all agents
                actions = self.get_actions(obs, add_noise=True, noise_scale=noise_scale)

                # Step environment
                next_obs, rewards, done, info = self.env.step(actions)

                # Store transition in replay buffer
                dones = [done] * self.n_agents  # Assuming shared done signal
                self.replay_buffer.add(obs, actions, rewards, next_obs, dones)

                # Update agents
                if step % self.update_every == 0 and step > 0:
                    actor_loss, critic_loss = self.update_agents(step)
                    if actor_loss is not None:
                        episode_actor_loss.append(actor_loss)
                        episode_critic_loss.append(critic_loss)

                obs = next_obs
                episode_reward += np.array(rewards)
                step += 1

                if done:
                    break

            # Log episode statistics
            self.episode_rewards.append(episode_reward)
            if episode_actor_loss:
                self.actor_losses.append(np.mean(episode_actor_loss))
                self.critic_losses.append(np.mean(episode_critic_loss))

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([np.sum(r) for r in self.episode_rewards[-10:]])
                print(f"Episode {episode + 1}/{total_episodes} | "
                      f"Avg Reward (last 10): {avg_reward:.2f} | "
                      f"Noise Scale: {noise_scale:.2f} | "
                      f"Buffer Size: {len(self.replay_buffer)}")

                if self.actor_losses:
                    print(f"  Actor Loss: {self.actor_losses[-1]:.4f} | "
                          f"Critic Loss: {self.critic_losses[-1]:.4f}")

            # Save model periodically
            if (episode + 1) % self.config['logging']['save_model_interval'] == 0:
                self.save_models(episode + 1)

        print("Training completed!")
        self.save_models(total_episodes)

    def save_models(self, episode):
        """Save all agent models."""
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        for i, agent in enumerate(self.agents):
            filepath = os.path.join(checkpoint_dir, f'agent_{i}_episode_{episode}.pth')
            agent.save(filepath)
        print(f"Models saved at episode {episode}")

    def load_models(self, episode):
        """Load all agent models."""
        checkpoint_dir = self.config['logging']['checkpoint_dir']

        for i, agent in enumerate(self.agents):
            filepath = os.path.join(checkpoint_dir, f'agent_{i}_episode_{episode}.pth')
            agent.load(filepath)
        print(f"Models loaded from episode {episode}")

    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate trained agents.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment

        Returns:
            avg_reward: Average total reward across episodes
        """
        print(f"Evaluating for {n_episodes} episodes...")

        episode_rewards = []

        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = np.zeros(self.n_agents)

            for t in range(self.max_episode_steps):
                if render:
                    self.env.render()

                # Get actions without noise
                actions = self.get_actions(obs, add_noise=False)

                # Step environment
                next_obs, rewards, done, info = self.env.step(actions)

                obs = next_obs
                episode_reward += np.array(rewards)

                if done:
                    break

            episode_rewards.append(np.sum(episode_reward))
            print(f"Episode {episode + 1}: Total Reward = {np.sum(episode_reward):.2f}")

        avg_reward = np.mean(episode_rewards)
        print(f"\nAverage Reward over {n_episodes} episodes: {avg_reward:.2f}")

        return avg_reward
