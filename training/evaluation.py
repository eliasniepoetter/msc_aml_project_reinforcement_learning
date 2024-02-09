from environment.envs.flight_env_target_altitude import FlightEnvTargetAltitude
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from custom_callback.successCallback import SuccessCallback
from stable_baselines3.common.callbacks import BaseCallback ,EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize


env = FlightEnvTargetAltitude()
model = PPO.load("./results/eval/models/best_model.zip")
model.set_env(env)


#this is for testing purposes
for i in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        # print(obs, rewards, done, info)
    env.render()
    print("Episode {} finished".format(i+1))