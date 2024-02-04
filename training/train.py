from environment.envs.flight_env_target_altitude import FlightEnvTargetAltitude
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from custom_callback.successCallback import SuccessCallback
from stable_baselines3.common.callbacks import BaseCallback ,EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
import time



# Initialize callback
# callback = SuccessCallback(check_freq=1) 
callback = StopTrainingOnRewardThreshold(reward_threshold=900) #currently not used

#initializing environment
env = FlightEnvTargetAltitude()
# env = VecNormalize(env)
env.reset()

# to use tensorboard log open terminal and run: tensorboard --logdir ./results/ppo_tensorboard/<subdirectory generated>
# TODO: add criteria to stop when 100 epoch has been successfull
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./results/ppo_tensorboard/").learn(total_timesteps=1000, progress_bar=True)

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
    # time.sleep(2)