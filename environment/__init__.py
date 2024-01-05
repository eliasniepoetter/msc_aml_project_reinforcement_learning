from gym.envs.registration import register

register(
    id='FlightEnv-v0',
    entry_point='environment.envs:FlightEnv',
    max_episode_steps=2000,
)