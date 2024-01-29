from environment.envs.flight_env import FlightEnv   
import numpy as np

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

        self.reward -= 1.005**(self.current_step) #! might be good to exclude

    def _get_target(self, atarget=None):
        target = np.random.uniform(200, 400)
        return target
    
    def _get_initialstate(self, a_state=None):
        state = np.array([0, 0, 0, 0, 0, np.random.uniform(200, 400)])
        initial_reward = 0
        return state, initial_reward
    
    #done condition is depreceated only use terminate and trunctuated
    def _EpisodeStopCondition(self, observation):
        if observation[-1] <= 0 or observation[-1] >= 600: # crashed into ground, or too much altitude #! use trunctuated
            done = True
        elif self.reward >= 1000: # successful episode with reward > 200 was able to keep altitude
            done = True
        elif self.current_step >= 1000: #! normaly regulated with max time steps in epoch
            done = True
        else:
            done = False
        
        return done