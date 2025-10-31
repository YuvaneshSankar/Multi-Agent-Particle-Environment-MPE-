import torch
import torch.nn.functional as F
import numpy as np
from maddpg.model import Actor, Critic


class MADDPGAgent:


    def __init__(self, agent_id, obs_dim, action_dim, n_agents, actor_lr, critic_lr,
                 tau, gamma, device, hidden_dim=128):

        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor network (policy) - uses only local observations
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic network (Q-function) - uses global state and all actions (centralized)
        # Input: all observations concatenated + all actions concatenated
        critic_input_dim = obs_dim * n_agents + action_dim * n_agents
        self.critic = Critic(critic_input_dim, hidden_dim).to(device)
        self.target_critic = Critic(critic_input_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize target networks with same weights as main networks
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def act(self, obs, noise=0.0, eval_mode=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy().flatten()
        self.actor.train()

        if not eval_mode:
            action += noise

        return np.clip(action, -1.0, 1.0)

    def target_act(self, obs):
        return self.target_actor(obs)

    def update_critic(self, obs_full, actions_full, rewards, next_obs_full,
                     next_actions_full, dones):
        # Compute current Q-value
        current_q = self.critic(obs_full, actions_full)

        # Compute target Q-value
        with torch.no_grad():
            target_q = self.target_critic(next_obs_full, next_actions_full)
            # Bellman backup: r + γ * Q_target(s', a') * (1 - done)
            y = rewards + self.gamma * target_q * (1 - dones)

        # Compute critic loss (MSE between current and target Q-values)
        critic_loss = F.mse_loss(current_q, y)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, obs_full, actions_full, agent_idx):
        # Reconstruct full action vector with this agent's action from current policy
        actions_for_q = torch.cat(actions_full, dim=1)

        # Policy gradient: maximize Q-value
        actor_loss = -self.critic(obs_full, actions_for_q).mean()

        # Add action regularization to prevent extreme actions
        actor_loss += (actions_full[agent_idx] ** 2).mean() * 1e-3

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_targets(self):
        """Soft update target networks."""
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def save(self, filepath):
        """Save agent's networks to file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load agent's networks from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
