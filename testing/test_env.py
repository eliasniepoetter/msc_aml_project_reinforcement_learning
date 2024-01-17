from environment.envs.flight_env import FlightEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np

env = FlightEnv()
check_env(env, warn=True, skip_render_check=True)
