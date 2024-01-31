from environment.envs.flight_env import FlightEnv   
import numpy as np

class FlightEnvTargetAltitude(FlightEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

    def _get_reward(self, observation,action):

        # define difference to target altitude
        difference_to_target = abs(observation[-1] - self.target_altitude)
        
        alpha = 1
        beta = 1
        self.reward = -alpha * difference_to_target - beta * np.linalg.norm(action)


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
        elif self.reward >= 1000: # successful episode with reward > 200 was able to keep altitude
            done = True
        elif self.current_step >= 10000: #! normaly regulated with max time steps in epoch
            done = True
        else:
            done = False
        
        return done