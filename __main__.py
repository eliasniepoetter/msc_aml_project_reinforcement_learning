from training import train

ppo_flight_env = train.PPOTraining()
ppo_flight_env.train(total_timesteps=1000000)
ppo_flight_env.evaluate(episode_count=5, model_name=None)
