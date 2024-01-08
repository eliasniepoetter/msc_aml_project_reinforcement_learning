from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from dynamics.flight_dynamics import FlightDynamics
from stable_baselines3.common.env_checker import check_env

class FlightEnv(Env):
    metadata = {'render_modes': ['human', 'flightgear'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        # Todo: define boundaries for action and observation space
        # Define observation and action space
        self.observation_space = Dict({'alpha':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                       'q':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                       'velocity':Box(low=0, high=1, shape=(1, 1), dtype=np.float32),
                                       'Theta':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                       'pos_x':Box(low=0, high=1, shape=(1, 1), dtype=np.float32),
                                       'pos_z':Box(low=0, high=1, shape=(1, 1), dtype=np.float32)})
        
        self.action_space = Dict({'elevator':Box(low=-1, high=1, shape=(1, 1), dtype=np.float32),
                                  'throttle':Box(low=0, high=1, shape=(1, 1), dtype=np.float32)})
        
        # Initial condition and state
        self.initial_state = np.array([0, 0, 100, 0, 0, 0]).T # ToDo: define initial state
        self.dynamics = FlightDynamics(initial_state=self.initial_state)
        self.current_step = 0
        self.dt = 0.1 # ToDo: define timestep
        self.reward = 0

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == 'flightgear':
            from visualization import FlightGearVisualizer
            self.visualizer = FlightGearVisualizer()

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
        state = self.dynamics.calculate_timestep(input=action, dt=self.dt)
        observation = self._state_to_observation(state)
        
        # ToDo: implement reward function
        reward = self.reward
        self.current_step += 1
        
        # ToDo: implement done condition
        done = self.current_step >= self.max_steps
        
        # ToDo: implement info
        info = {}

        return observation, reward, done, info

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
        # ToDo: normalize state
        observation = {'alpha':state[0],
                       'q':state[1],
                       'velocity':state[2],
                       'Theta':state[3],
                       'pos_x':state[4],
                       'pos_z':state[5]}
        return observation
   