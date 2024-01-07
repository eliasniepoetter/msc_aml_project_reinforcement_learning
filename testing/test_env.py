from environment.envs.flight_env import FlightEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np

env = FlightEnv()
#check_env(env, warn=True, skip_render_check=True)
sample = env.observation_space.sample()
print(sample, '\n')
state = np.array([sample['alpha'], sample['q'], sample['velocity'], sample['Theta'], sample['pos_x'], sample['pos_z']]).reshape(6,1)
print(state, '\n')
print(env._state_to_observation(state), '\n')