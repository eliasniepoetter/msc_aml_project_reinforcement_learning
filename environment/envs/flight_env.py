from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from dynamics.flightdynamics import Flightdynamics
from stable_baselines3.common.env_checker import check_env

class FlightEnv(Env):
    metadata = {'render_modes': ['human', 'flightgear'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        # Todo: define boundaries for action and observation space, append observation space with target altitude
        # Define observation and action space
        '''
        self.observation_space = Dict({'alpha':Box(low=-0.2, high=0.2, shape=(1,), dtype=np.float64),
                                       'q':Box(low=-2, high=2, shape=(1,), dtype=np.float64),
                                       'velocity':Box(low=-100, high=100, shape=(1,), dtype=np.float64),
                                       'Theta':Box(low=-1.2, high=1.2, shape=(1,), dtype=np.float64),
                                       'pos_x':Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
                                       'pos_z':Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)})
        '''
        # observation space [alpha, q, velocity, Theta, pos_x, pos_z]
        self.observation_space = Box(low=np.array([-0.2, -2, -100, -1.2, 0, 0]), high=np.array([0.2, 2, 100, 1.2, np.inf, np.inf]), shape=(6,), dtype=np.float64)
        
        '''
        self.action_space = Dict({'elevator':Box(low=-0.7, high=0.7, shape=(1,), dtype=np.float64),
                                  'throttle':Box(low=0, high=1, shape=(1,), dtype=np.float64)})
        '''
        # action space [elevator, throttle]
        self.action_space = Box(low=np.array([-0.7, 0]), high=np.array([0.7, 1]), shape=(2,), dtype=np.float64)
        
        # Initial condition and state
        self.initial_state = np.array([0, 0, 0, 0, 0, np.random.uniform(200, 400)]) # ToDo: define initial state
        self.dynamics = Flightdynamics(initial_state=self.initial_state)
        self.current_step = 0
        self.dt = 0.01 # ToDo: define timestep
        
        # try to hold altitude
        self.target_altitude = np.random.uniform(200, 400)        
        self.reward = 0

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.reward = 0

        # Todo
        self.target_altitude = np.random.uniform(200, 400) 
        # set state of dynamics to initial state and with new random altitude convert to observation
        self.initial_state[-1] = np.random.uniform(200, 400) 
        self.dynamics.state = self.initial_state

        observation = self.initial_state

        info = {}

        return observation, info
    
    def step(self, action):
        observation = self.dynamics.timestep(input=action, dt=self.dt)
        
        # ToDo: implement reward function
        self.current_step += 1
        self.reward += 1     # get reward for surviving
        delta_h = observation[-1] - self.target_altitude     
        if delta_h < 10:
            self.reward += 100   # get reward for being close to target altitude
        
        # ToDo: implement done condition
        done = False
        truncated = False

        # ToDo: implement info, can be empty
        info = {}

        return observation, self.reward, done, truncated, info

    def render(self):
        # ToDo: implement flight visualization with flightgear
        if self.render_mode == 'human':
            pass

    def close(self):
        print('close')
        pass
    
    def reward_function(self, state):
        raise NotImplementedError