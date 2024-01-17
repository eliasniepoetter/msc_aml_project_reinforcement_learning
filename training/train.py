from environment.envs.flight_env import FlightEnv
from stable_baselines3 import PPO

env = FlightEnv()
model = PPO('MlpPolicy', env, verbose=1).learn(100)