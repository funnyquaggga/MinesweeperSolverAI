import gymnasium as gym
import numpy as np

class ActionMaskEnv(gym.Wrapper):
    def __init__(self, env):
        super(ActionMaskEnv, self).__init__(env)
        self.valid_actions = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._update_valid_actions()
        return observation, info

    def step(self, action):
        if action not in self.valid_actions:
            # Penalize invalid actions
            reward = -0.1
            terminated = False
            truncated = False
            observation = self.env._get_observation()
            info = {'invalid_action': True}
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._update_valid_actions()
        if np.isscalar(observation):
            observation = np.array([observation], dtype=np.float32)
        print(f"Observation type: {type(observation)}, shape: {observation.shape}, dtype: {observation.dtype}")
        return observation, reward, terminated, truncated, info

    def _update_valid_actions(self):
        # Update the list of valid actions (indices of unrevealed cells)
        self.valid_actions = np.flatnonzero(~self.env.revealed.flatten())
