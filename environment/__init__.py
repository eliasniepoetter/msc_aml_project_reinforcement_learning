from gymnasium.envs.registration import register

register(
     id="environment/FlightEnv-v0",
     entry_point="environment.envs:FlightEnv",
     max_episode_steps=300,
)