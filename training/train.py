from environment.envs.flight_env import FlightEnv
from stable_baselines3 import PPO

env = FlightEnv()
env.reset()
model = PPO('MlpPolicy', env, verbose=1).learn(1000)

for i in range(1000):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        print(obs, rewards, done, info)
    env.render()
    print("Episode {} finished".format(i+1))
    input("Press [enter] to continue.")