from training import train

ppo_flight_env = train.PPOTraining()
#ppo_flight_env.train(total_timesteps=10000000)
#! change model name to evaluate different model, or None to evaluate current model
ppo_flight_env.evaluate(episode_count=5, model_name=r"results\eval\models\best_model.zip") 
