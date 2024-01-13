from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from dynamics.flightdynamics import FlightDynamics
from stable_baselines3.common.env_checker import check_env

class FlightEnv(Env):
    metadata = {'render_modes': ['human', 'flightgear'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        # Todo: define boundaries for action and observation space, append observation space with target altitude
        # Define observation and action space
        self.observation_space = Dict({'alpha':Box(low=-0.2, high=0.2, shape=(1, 1), dtype=np.float32),
                                       'q':Box(low=-2, high=2, shape=(1, 1), dtype=np.float32),
                                       'velocity':Box(low=-100, high=100, shape=(1, 1), dtype=np.float32),
                                       'Theta':Box(low=-1.2, high=1.2, shape=(1, 1), dtype=np.float32),
                                       'pos_x':Box(low=0, high=np.inf, shape=(1, 1), dtype=np.float32),
                                       'pos_z':Box(low=0, high=np.inf, shape=(1, 1), dtype=np.float32)})
        
        self.action_space = Dict({'elevator':Box(low=-0.7, high=0.7, shape=(1, 1), dtype=np.float32),
                                  'throttle':Box(low=0, high=1, shape=(1, 1), dtype=np.float32)})
        
        # Initial condition and state
        self.initial_state = np.array([0, 0, 0, 0, 0, np.random.uniform(200, 400)]).T # ToDo: define initial state
        self.dynamics = FlightDynamics(initial_state=self.initial_state)
        self.current_step = 0
        self.dt = 0.01 # ToDo: define timestep
        
        # try to hold altitude
        self.target_altitude = np.random.uniform(200, 400)        
        self.reward = 0

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == 'flightgear':
            from visualization import FlightGearVisualizer
            self.visualizer = FlightGearVisualizer()

    def reset(self):
        self.current_step = 0
        self.reward = 0
        self.target_altitude = np.random.uniform(200, 400) 
        # set state of dynamics to initial state and with new random altitude convert to observation
        self.initial_state[-1] = np.random.uniform(200, 400) 
        self.dynamics.state = self.initial_state

        observation = self._state_to_observation(self.initial_state)

        return observation
    
    def step(self, action):
        input = self._action_to_input(self, action)
        state = self.dynamics.calculate_timestep(input=input, dt=self.dt)
        observation = self._state_to_observation(state)
        
        # ToDo: implement reward function
        self.current_step += 1
        self.reward += 1     # get reward for surviving
        delta_h = state[-1] - self.target_altitude     
        if delta_h < 10:
            self.reward += 100   # get reward for being close to target altitude
        
        # ToDo: implement done condition
        done = self.current_step >= self.max_steps
        
        # ToDo: implement info
        info = {}

        return observation, self.reward, done, info

    def render(self):
        # ToDo: implement flight visualization with flightgear
        if self.render_mode == 'human':
            pass
        elif self.render_mode == 'flightgear':
            pass

    def close(self):
        print('close')
        pass

    def _state_to_observation(self, state):
        observation = {'alpha':state[0],
                       'q':state[1],
                       'velocity':state[2],
                       'Theta':state[3],
                       'pos_x':state[4],
                       'pos_z':state[5]}
        return observation
    
    def _action_to_input(self, action):
        return np.array([[action['elevator']], [action['throttle']]])
        
   