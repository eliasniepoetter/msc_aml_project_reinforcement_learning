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
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os

N_ENVS = os.cpu_count()

class PPOTraining():
    def __init__(self):
        # Initialize callback
        # self.callback = SuccessCallback(check_freq=1) 
        #initializing environment
        self.env = FlightEnvTargetAltitude()
        # self.env = VecNormalize(env)
        self.model = None
    
    def train(self, total_timesteps=1000000):
        # wrapper for environment for callback and monitoring
        self.env = Monitor(self.env, filename='./results/monitor', allow_early_resets=True)

        # reset at the before start of training
        self.env.reset()

        callback_stop = StopTrainingOnRewardThreshold(reward_threshold=2e7) #currently not used
        callback = EvalCallback(self.env, callback_after_eval=callback_stop, 
                                best_model_save_path='./results/eval/models/', log_path='./results/eval/logs/',
                                eval_freq=100000, deterministic=True, render=True, n_eval_episodes=10, verbose=1)

        # to use tensorboard log open terminal and run: tensorboard --logdir ./results/ppo_tensorboard/<subdirectory generated>
        # TODO: add criteria to stop when 100 epoch has been successfull
        self.model = PPO('MlpPolicy', self.env, verbose=1, ent_coef=0.02, batch_size=1024, tensorboard_log="./results/ppo_tensorboard/").learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)


        rewards_list = self.env.get_episode_rewards()
        episode_length = self.env.get_episode_lengths()


        # evalute the policy

        # if other model than current model should be evaluated, load the model and set the environment
        # model = PPO.load("ppo_flight_env")
        # model.set_env(env)

        episode_reward_list, episode_length_list = evaluate_policy(self.model, self.env, n_eval_episodes=1, render=True, deterministic=True, return_episode_rewards=True, warn=True)

    def evaluate(self, episode_count=5, model_name=None):
        # load model if model already exists
        if model_name is not None:
            self.model = PPO.load(model_name)

        for i in range(episode_count):
            obs, info = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, rewards, done, _, info = self.env.step(action)
            self.env.render()
            print("Episode {} finished".format(i+1))
