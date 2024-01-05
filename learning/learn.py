from flight_env import FlightEnv

env = FlightEnv()
episodes = 100

for episode in range(episodes):
    done = False
    obs = env.reset()
    for step in range(200):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode {} finished after {} timesteps".format(episode, step+1))
            break