from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from dynamics.flightdynamics import Flightdynamics
from stable_baselines3.common.env_checker import check_env
from collections import deque
from environment.visualization import PlotVisualizer
from abc import ABC, abstractmethod

class FlightEnv(Env,ABC):
    #metadata = {'render_modes': ['human'], 'render_fps': 30}

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
        self.memory_size = 10
        state_lower_bound = np.array([-0.2, -2, -100, -1.2, 0, 0])
        state_upper_bound = np.array([0.2, 2, 100, 1.2, np.inf, np.inf])
        self.observation_space = Box(low=np.repeat(state_lower_bound, self.memory_size), high=np.repeat(state_upper_bound, self.memory_size), shape=(6*self.memory_size,), dtype=np.float64)
        
        # Initialize the deque to store state history
        self.state_memory = deque(maxlen=self.memory_size)

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
        self.obs_act_collection = deque()
        
        # try to hold altitude
        self.target_altitude = np.random.uniform(200, 400)        
        self.reward = 0

        # rendering
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode = render_mode
        self.vis = PlotVisualizer(self.target_altitude)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.reward = 0
        self.vis.reset_plot()

        # Reset the environment and clear the state memory
        self.state_memory.clear()

        # Todo
        self.target_altitude = np.random.uniform(200, 400) 
        # set state of dynamics to initial state and with new random altitude convert to observation
        self.initial_state[-1] = np.random.uniform(200, 400) 
        self.dynamics.state = self.initial_state
        self.obs_act_collection.clear()

        # Add the initial state to the state memory
        self.state_memory.extend([self.initial_state] * self.memory_size)

        observation = np.concatenate(self.state_memory, axis=-1)

        info = {}

        return observation, info
    
    def step(self, action):
        # observation = self.dynamics.timestep(input=action, dt=self.dt)
        obs = self.dynamics.timestep(input=action, dt=self.dt)
        # save observation and action in collection
        self.obs_act_collection.append(np.concatenate((obs, action)))

        # Add the current observation to the state memory
        self.state_memory.append(obs)

        # Concatenate the state memory to form the agent's current state
        # newest obsevation is last vector in concatenated observation!
        observation = np.concatenate(self.state_memory, axis=-1)

        # ToDo: implement reward function
        self.current_step += 1
        self._get_reward(observation)
        
        # ToDo: implement done condition
        if self.current_step >= 5000:
            done = True
        elif observation[-1] <= 0:
            done = True
        else:
            done = False
        truncated = False

        # ToDo: implement info, can be empty
        info = {}

        return observation, self.reward, done, truncated, info

    def render(self):
        #if self.render_mode == 'human':
            #pass
        self.vis.render_state(self.obs_act_collection, self.dt, self.target_altitude)

    def close(self):
        self.vis.close()
    
    @abstractmethod
    def _get_reward(self, observation):
        raise NotImplementedError