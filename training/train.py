from environment.envs.flight_env_target_altitude import FlightEnvTargetAltitude
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

#initializing environment
env = FlightEnvTargetAltitude()

# wrapper for environment for callback and monitoring
env = Monitor(env, filename='./results/monitor', allow_early_resets=True)

# callback for maximum mean reward
callback_stop = StopTrainingOnRewardThreshold(reward_threshold=2e7)
callback = EvalCallback(env, callback_after_eval=callback_stop, 
                        best_model_save_path='./results/eval/models/', log_path='./results/eval/logs/',
                          eval_freq=100000, deterministic=True, render=True, n_eval_episodes=10, verbose=1)

# train the model with PPO over total_timesteps=xxx timesteps
# to use tensorboard log open terminal and run: tensorboard --logdir ./results/ppo_tensorboard/<subdirectory generated>
model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.02, batch_size=1024, tensorboard_log="./results/ppo_tensorboard/").learn(total_timesteps=10, progress_bar=True, callback=callback)

rewards_list = env.get_episode_rewards()
episode_length = env.get_episode_lengths()

# evalute the policy
episode_reward_list, episode_length_list = evaluate_policy(model, env, n_eval_episodes=1, render=True, deterministic=True, return_episode_rewards=True, warn=True)
for i in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
    env.render()
    print("Episode {} finished".format(i+1))

env.close()