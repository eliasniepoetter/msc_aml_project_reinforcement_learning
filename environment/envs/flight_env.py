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
        # Define observation and action space
        # observation space [alpha, q, velocity, Theta, pos_x, pos_z, target_altitude]
        self.memory_size = 10
        state_lower_bound = np.array([-0.2, -0.2, -30, -1.0, 0, 0, 0])
        state_upper_bound = np.array([0.2, 0.2, 30, 1.0, np.inf, np.inf, np.inf])
        self.observation_space = Box(low=np.repeat(state_lower_bound, self.memory_size), high=np.repeat(state_upper_bound, self.memory_size), shape=(7*self.memory_size,), dtype=np.float64)
        # Initialize the deque to store observation history
        self.state_memory = deque(maxlen=self.memory_size)

        # action space [elevator, throttle]
        self.action_space = Box(low=np.array([-0.1, -0.5]), high=np.array([0.1, 0.5]), shape=(2,), dtype=np.float64)
        
        # Initial condition and state
        self.target_altitude = self._get_target()
        self.initial_state, self.reward  = self._get_initialstate()
        # Add the initial state to the state memory
        self.state_memory.extend([self.initial_state] * self.memory_size)
        self.dynamics = Flightdynamics()
        self.current_step = 0
        self.dt = 0.05
        self.obs_act_collection = deque()
        
        # rendering
        self.vis = PlotVisualizer() 

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed, options=options)
        self.current_step = 0
        
        # Reset the environment and clear the state memory
        self.state_memory.clear()

        self.target_altitude = self._get_target() 
        # set state of dynamics to initial state and with new random altitude convert to observation
        self.initial_state, self.reward = self._get_initialstate()
        self.obs_act_collection.clear()

        # Add the initial state to the state memory
        self.state_memory.extend([self.initial_state] * self.memory_size)

        # Concatenate the state memory to form the agent's current state. Newest observation is last vector in concatenated observation
        observation = np.concatenate(self.state_memory, axis=-1)

        info = {}

        return observation, info
    
    def step(self, action):
        # action rate limiter
        self.rateLimitElevator = 0.1 # rad/s
        self.rateLimitThrottle = 1 # 1/s
        if self.current_step > 0:
            if abs(action[0] - self.obs_act_collection[-1][-2]) > self.rateLimitElevator*self.dt:
                sign = np.sign(action[0] - self.obs_act_collection[-1][-2])
                action[0] = self.obs_act_collection[-1][-2] + sign*self.rateLimitElevator*self.dt

            if abs(action[1] - self.obs_act_collection[-1][-1]) > self.rateLimitThrottle*self.dt:
                sign = np.sign(action[1] - self.obs_act_collection[-1][-1])
                action[1] = self.obs_act_collection[-1][-1] + sign*self.rateLimitThrottle*self.dt
        elif self.current_step == 0:
            action = np.array([0, 0]) # initial actions for first step

        # get observation from dynamics
        obs = self.dynamics.timestep(current_state=self.state_memory[-1], input=action, dt=self.dt)
        # Add observation and action in collection for episode visualization
        self.obs_act_collection.append(np.concatenate((obs, action)))
        # Add the current observation to the state memory
        self.state_memory.append(obs)
        # Concatenate the state memory to form the agent's current state. Newest observation is last vector in concatenated observation
        observation = np.concatenate(self.state_memory, axis=-1)

        # increment step counter and get step reward
        self.current_step += 1
        self._get_reward(observation,action)
        
        # check if episode is done
        done = self._EpisodeStopCondition(observation=observation)
        
        # nessessary for gymnasium environment
        truncated = False
        info = {}

        return observation, self.reward, done, truncated, info

    def render(self):
        self.vis.visualize_episode(self.obs_act_collection, self.dt, self.target_altitude)

    def close(self):
        self.vis.close()
    
    @abstractmethod
    def _get_reward(self, observation):
        raise NotImplementedError
    
    @abstractmethod
    # returns initial state of aircraft
    def _get_initialstate(self, a_state=None):
        raise NotImplementedError
    
    @abstractmethod
    #returns desired target used for reward function 
    def _get_target(self, atarget=None):
        raise NotImplementedError
    
    @abstractmethod
    def _EpisodeStopCondition(self, observation):
        raise NotImplementedError