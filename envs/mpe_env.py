import numpy as np
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios import load as load_scenario

class MPEEnvWrapper:
    def __init__(self, scenario_name="simple_spread", max_episode_steps=25, seed=None):
        self.scenario = load_scenario(scenario_name + ".py")  # Load scenario file
        self.env = MultiAgentEnv(self.scenario, reset_callback=None, reward_callback=None, observation_callback=None, info_callback=None,
                                 shared_viewer=False)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.current_step = 0
        obs = self.env.reset()
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        self.current_step += 1
        done = all(dones) or self.current_step >= self.max_episode_steps
        return obs, rewards, done, infos

    def render(self):
        self.env.render()

    @property
    def observation_space(self):
        return [agent.observation_space for agent in self.env.agents]

    @property
    def action_space(self):
        return [agent.action_space for agent in self.env.agents]
