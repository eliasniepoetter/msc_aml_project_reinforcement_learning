from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from dynamics import Dynamics

class FlightEnv(Env):

    def __init__(self):
        # Todo: define boundaries for action and observation space
        # defintion of spaces
        self.observation_space = Dict({'velocity':Box(low=0, high=1, shape=(1, 1), dtype=np.float32),
                                       'alpha':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                       'pos_z':Box(low=0, high=1, shape=(1, 1), dtype=np.float32),
                                       'pos_x':Box(low=0, high=1, shape=(1, 1), dtype=np.float32),
                                       'q':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                       'Theta':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)})
        self.action_space = Dict({'elevator':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                  'throttle':Box(low=0, high=1, shape=(1, 1), dtype=np.float32)})
        
        # initial condition and state
        self.reset()

    def reset(self):
        # ToDo: set dynamics to initial state
        self.current_step = 0
        self.reward = 0
        initial_state = None # ToDo: define initial state
        
        # check if reset is called for the first time
        if not self.dynamics in locals():
            self.dynamics = Dynamics()

        # set state of dynamics to initial state and convert to observation
        self.dynamics.state = initial_state
        observation = self._state_to_observation(initial_state)

        return observation
    
    def step(self, action):
        # ToDo: convert action to input for dynamics and observation from output of dynamics
        state = Dynamics.updateStates(action,printState=False)
        observation = self._state_to_observation(state)
        
        # ToDo: implement reward function
        reward = self.reward
        self.current_step += 1
        
        # ToDo: implement termination condition
        terminated = False
        
        # ToDo: implement done condition
        done = False
        
        return observation, reward, terminated, done, info

    def render(self):
        # ToDo: implement flight visualization with flightgear
        pass

    def close(self):
        pass

    def _state_to_observation(self, state):
        # ToDo: implement conversion from state to observation based on observation space and state definition
        pass