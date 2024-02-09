from environment.envs.flight_env_target_altitude import FlightEnvTargetAltitude
from stable_baselines3 import PPO

env = FlightEnvTargetAltitude()
model = PPO.load("./results/eval/models/best_model.zip")
model.set_env(env)

# evalute the best model
for i in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
    env.render()
    print("Episode {} finished".format(i+1))