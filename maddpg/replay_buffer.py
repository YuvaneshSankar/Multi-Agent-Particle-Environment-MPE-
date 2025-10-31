import numpy as np
import random
from collections import deque


class ReplayBuffer:


    def __init__(self, buffer_size, n_agents):
        """
        Initialize replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            n_agents: Number of agents in the environment
        """
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs, actions, rewards, next_obs, dones):
        """
        Add a transition to the buffer.

        Args:
            obs: List of observations for each agent
            actions: List of actions taken by each agent
            rewards: List of rewards received by each agent
            next_obs: List of next observations for each agent
            dones: List of done flags for each agent
        """
        # Store as a single transition tuple
        experience = (obs, actions, rewards, next_obs, dones)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            obs_batch: List of observation batches for each agent (n_agents, batch_size, obs_dim)
            actions_batch: List of action batches for each agent (n_agents, batch_size, action_dim)
            rewards_batch: List of reward batches for each agent (n_agents, batch_size, 1)
            next_obs_batch: List of next observation batches for each agent (n_agents, batch_size, obs_dim)
            dones_batch: List of done flag batches for each agent (n_agents, batch_size, 1)
        """
        # Sample random batch of experiences
        batch = random.sample(self.buffer, batch_size)

        # Initialize lists for each component
        obs_batch = [[] for _ in range(self.n_agents)]
        actions_batch = [[] for _ in range(self.n_agents)]
        rewards_batch = [[] for _ in range(self.n_agents)]
        next_obs_batch = [[] for _ in range(self.n_agents)]
        dones_batch = [[] for _ in range(self.n_agents)]

        # Organize batch by agent
        for experience in batch:
            obs, actions, rewards, next_obs, dones = experience

            for i in range(self.n_agents):
                obs_batch[i].append(obs[i])
                actions_batch[i].append(actions[i])
                rewards_batch[i].append([rewards[i]])
                next_obs_batch[i].append(next_obs[i])
                dones_batch[i].append([dones[i]])

        # Convert to numpy arrays
        obs_batch = [np.array(obs_batch[i]) for i in range(self.n_agents)]
        actions_batch = [np.array(actions_batch[i]) for i in range(self.n_agents)]
        rewards_batch = [np.array(rewards_batch[i]) for i in range(self.n_agents)]
        next_obs_batch = [np.array(next_obs_batch[i]) for i in range(self.n_agents)]
        dones_batch = [np.array(dones_batch[i]) for i in range(self.n_agents)]

        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size
