from environment.envs.flight_env import FlightEnv   
import numpy as np

class FlightEnvTargetAltitude(FlightEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

    def _get_reward(self, observation, action):

        self.reward = 0

        # define difference to target altitude
        difference_to_target = abs(observation[-2] - self.target_altitude)
        difference_to_target_span = abs(observation[5] - self.target_altitude)
        
        V0 = 51.4

        # reward function parameters
        alpha = 0.01
        beta = 0.001
        survival_factor = 0.05

        survival_reward = self.current_step * survival_factor

        v_current = observation[-5]
        
        if abs(v_current) < 0.9*V0 or abs(v_current) > 1.1*V0:
            velocity_reward = -10
        else:
            velocity_reward = 0


        if difference_to_target_span > difference_to_target:
            directional_reward = 1
        else:
            directional_reward = -10

        if difference_to_target < 10:
            bonus = 100
        else:
            bonus = 0

        # simple reward function
        self.reward = -alpha * difference_to_target - beta * np.linalg.norm(action) + survival_reward + directional_reward + velocity_reward + bonus

        # hard punishment for attempting to simulate the Boeing 737 Max altitude controller (MKAS)
        if observation[-2] <= 00:
            self.reward = -1e6
        elif observation[-2] >= 1000:
            self.reward = -1e6

    def _get_target(self, atarget=None):
        target = np.random.uniform(200, 400)
        return target
    
    def _get_initialstate(self, a_state=None):
        state = np.array([0, 0, 0, 0, 0, np.random.uniform(200, 400), self.target_altitude])
        initial_reward = 0
        return state, initial_reward
    
    #done condition is depreceated only use terminate and trunctuated
    def _EpisodeStopCondition(self, observation):
        if observation[-2] <= 0 or observation[-2] >= 1000: # crashed into ground, or too much altitude #! use trunctuated
            done = True
        elif self.current_step >= 10000: #! normaly regulated with max time steps in epoch
            done = True
        else:
            done = False
        
        return done