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
        
        # Initialize the deque to store state history
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
        self.dt = 0.05 # ToDo: define timestep
        self.obs_act_collection = deque()
        
        # rendering
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode = render_mode
        self.vis = PlotVisualizer(self.target_altitude) 

        #counter for success
        self.success_count = 0

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.vis.reset_plot()

        # Reset the environment and clear the state memory
        self.state_memory.clear()

        self.target_altitude = self._get_target() 
        # set state of dynamics to initial state and with new random altitude convert to observation
        self.initial_state, self.reward = self._get_initialstate()
        self.obs_act_collection.clear()

        # Add the initial state to the state memory
        self.state_memory.extend([self.initial_state] * self.memory_size)

        observation = np.concatenate(self.state_memory, axis=-1)

        info = {}

        return observation, info
    
    def step(self, action):
        # observation = self.dynamics.timestep(input=action, dt=self.dt)

        # initial actions for first step
        if self.current_step == 0:
            action = np.array([0, 0])

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

        obs = self.dynamics.timestep(current_state=self.state_memory[-1], input=action, dt=self.dt)
        # save observation and action in collection
        self.obs_act_collection.append(np.concatenate((obs, action)))

        # Add the current observation to the state memory
        self.state_memory.append(obs)

        # Concatenate the state memory to form the agent's current state
        # newest observation is last vector in concatenated observation!
        observation = np.concatenate(self.state_memory, axis=-1)

        self.current_step += 1
        #self._get_reward(observation,action)
        self._get_simple_reward(observation,action)
        
        done = self._EpisodeStopCondition(observation=observation)
        
        #test purposes
        # if done == True:
            # print(self.current_step)
            # print(self.reward)
            # print(observation)
            # print(self.target_altitude)

        truncated = False

        # ToDo: implement info, can be empty
        info = {}

        return observation, self.reward, done, truncated, info

    def render(self):
        #if self.render_mode == 'human':
            #pass
        #self.vis.render_state(self.obs_act_collection, self.dt, self.target_altitude)
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