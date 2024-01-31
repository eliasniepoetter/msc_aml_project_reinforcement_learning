from environment.envs.flight_env import FlightEnv   
import numpy as np

class FlightEnvTargetAltitude(FlightEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

    def _get_reward(self, observation,action):

        self.reward = 0

        # define difference to target altitude
        difference_to_target = abs(observation[-1] - self.target_altitude)
        difference_to_target_span = abs(observation[5] - self.target_altitude)
        
        # reward function parameters
        alpha = 0.001
        beta = 0.01
        survival_factor = 0.5

        survival_reward = self.current_step * survival_factor

        if difference_to_target_span > difference_to_target:
            directional_reward = 1
        else:
            directional_reward = -10

        # simple reward function
        self.reward = -alpha * difference_to_target - beta * np.linalg.norm(action) + survival_reward + directional_reward

        # hard punishment for attempting to simulate the Boeing 737 Max altitude controller (MKAS)
        if observation[-1] <= 00:
            self.reward = -1e6
        elif observation[-1] >= 1000:
            self.reward = -1e6

    def _get_target(self, atarget=None):
        target = np.random.uniform(200, 400)
        return target
    
    def _get_initialstate(self, a_state=None):
        state = np.array([0, 0, 0, 0, 0, np.random.uniform(200, 400)])
        initial_reward = 0
        return state, initial_reward
    
    #done condition is depreceated only use terminate and trunctuated
    def _EpisodeStopCondition(self, observation):
        if observation[-1] <= 0 or observation[-1] >= 1000: # crashed into ground, or too much altitude #! use trunctuated
            done = True
        elif self.current_step >= 10000: #! normaly regulated with max time steps in epoch
            done = True
        else:
            done = False
        
        return done