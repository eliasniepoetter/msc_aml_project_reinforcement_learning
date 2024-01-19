from environment.envs.flight_env_target_altitude import FlightEnvTargetAltitude
from stable_baselines3 import PPO
import time

env = FlightEnvTargetAltitude()
env.reset()
model = PPO('MlpPolicy', env, verbose=100).learn(10000)

for i in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        print(obs, rewards, done, info)
    env.render()
    print("Episode {} finished".format(i+1))
    time.sleep(2)