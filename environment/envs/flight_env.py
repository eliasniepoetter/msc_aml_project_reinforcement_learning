from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from dynamics.flight_dynamics import Dynamics
from stable_baselines3.common.env_checker import check_env

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
        self.initial_state = np.array([0, 0, 100, 0]) # ToDo: define initial state
        self.dynamics = Dynamics(id='1',state=self.initial_state)
        self.current_step = 0
        self.reward = 0

    def reset(self):
        # ToDo: set dynamics to initial state
        self.current_step = 0
        self.reward = 0
        # set state of dynamics to initial state and convert to observation
        self.dynamics.state = self.initial_state
        observation = self._state_to_observation(self.initial_state)

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
        
        # ToDo: implement info
        info = 'info'
        return observation, reward, terminated, done, info

    def render(self):
        # ToDo: implement flight visualization with flightgear
        pass

    def close(self):
        print('close')
        pass

    def _state_to_observation(self, state):
        # ToDo: implement conversion from state to observation based on observation space and state definition
        pass
   