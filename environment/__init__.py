from gymnasium.envs.registration import register

register(
     id="environment/FlightEnv-v0",
     entry_point="environment.envs:FlightEnv",
)