from environment.envs.flight_env import FlightEnv   
import numpy as np

class FlightEnvTargetAltitude(FlightEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

    def _get_reward(self, observation, action):
        # reset step reward
        self.reward = 0
        
        # decompose observations for better readability
        v_z = observation[1]
        dz = observation[-1]

        # reward function parameters
        k_altitude = 0.01           # altitude reward factor
        k_survival = 0              # survival reward factor, included in altitude reward
        k_direction = 0             # direction reward factor
        k_actuator_effort = 0.001   # actuator effort reward factor

        altitude_reward = np.exp(-abs(dz) * k_altitude)
        survival_reward = self.current_step * k_survival
        actuator_effort_reward = -k_actuator_effort * np.linalg.norm(action)
        if -v_z*dz >= 0:
            direction_reward = k_direction
        else:
            direction_reward = -k_direction

        # calculate total step reward
        self.reward = altitude_reward + actuator_effort_reward + survival_reward + direction_reward
        
    def _get_target(self, atarget=None):
        target = np.random.uniform(200, 400)
        return target
    
    def _get_initial_observation(self, a_state=None):
        z_initial = np.random.uniform(200, 400)
        initial_observation = np.array([0, 0, 0, z_initial, z_initial-self.target_altitude])
        initial_reward = 0
        return initial_observation, initial_reward
    
    #done condition is depreceated only use terminate and trunctuated
    def _EpisodeStopCondition(self, observation):
        if observation[-2] <= 0 or observation[-2] >= 1000: # crashed into ground, or too much altitude #! use trunctuated
            done = True
        elif self.current_step >= 10000: #! normaly regulated with max time steps in epoch
            done = True
        else:
            done = False

        truncated = False
        return done, truncated