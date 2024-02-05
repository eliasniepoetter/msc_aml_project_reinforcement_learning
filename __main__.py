from training import train

ppo_flight_env = train.PPOTraining()
#ppo_flight_env.train()
ppo_flight_env.evaluate(episode_count=5, model_name="results/eval/models/best_model.zip")
