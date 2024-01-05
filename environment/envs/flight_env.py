import gymnasium as gym
from gymnasium import spaces

class FlightEnv(gym.Env):
    def __init__(self):
        # Todo: define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)

    def step(self, action):
        pass
        return observation, reward, terminated, done, info

    def reset(self):
        pass
        return observation

    def render(self):
        pass

    def close(self):
        pass