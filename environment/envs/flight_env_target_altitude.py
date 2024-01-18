from environment.envs.flight_env import FlightEnv   

class FlightEnvTargetAltitude(FlightEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

    def _get_reward(self, observation):
        # define difference to target altitude
        overallSize = self.memory_size*6
        difference_to_target = abs(observation[-1] - self.target_altitude)
        difference_to_target_last10 = abs(observation[overallSize-(6*(self.memory_size-1)-1)] - self.target_altitude)

        if difference_to_target_last10-difference_to_target > 0:
            self.reward += 1
        else:
            self.reward -= 10

        if difference_to_target < 10:
            self.reward += 100

        self.reward -= 1.005**(self.current_step)