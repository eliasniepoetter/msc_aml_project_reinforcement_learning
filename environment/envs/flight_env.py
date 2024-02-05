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
        # observation and action space definition
        # ToDo: add state memory if needed
        # observation space             [v_x,   v_y,    x,      z,      dz      ]
        state_lower_bound = np.array(   [-50,   -50,    0,      0,      0       ])
        state_upper_bound = np.array(   [50,    50,     np.inf, np.inf, np.inf  ])
        self.observation_space = Box(low=state_lower_bound, high=state_upper_bound, shape=(len(state_lower_bound),), dtype=np.float64)

         # action space                 [elevator,  throttle]
        action_lower_bound = np.array(  [-0.1,      -0.5    ])
        action_upper_bound = np.array(  [0.1,       0.5     ])                            
        self.action_space = Box(low=action_lower_bound, high=action_upper_bound, shape=(len(action_lower_bound),), dtype=np.float64)
        
        # initial condition and state
        self.target_altitude = self._get_target()
        self.observation, self.reward  = self._get_initialstate()

        # initialize dynamics
        # Todo: define timestep
        self.dt = 0.05 
        self.dynamics = Flightdynamics()
        
        # initialize memory
        self.obs_act_collection = deque()
        
        # initialize progress tracking
        self.current_step = 0
        self.success_count = 0

        # initialize rendering
        # ToDO: implement rendering
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode = render_mode
        self.vis = PlotVisualizer(self.target_altitude) 
    
    def reset(self, seed=None, options=None): 
        super().reset(seed=seed, options=options)

        # reset progress tracking
        self.current_step = 0
        self.success_count = 0       

        # Reset the environment and clear the state memory
        self.obs_act_collection.clear()

        # get new target altitude and initialize state
        self.target_altitude = self._get_target()
        self.observation, self.reward = self._get_initialstate()
        self.dynamics.reset()

        # reset visualization
        self.vis.reset_plot()

        # ToDo: implement info, can be empty
        info = {}

        return self.observation, info
    
    def step(self, action):
        # action rate limiter and initial action
        # ToDo: adapt rate limit
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
            action = np.array([0, 0])

        # get new observation
        state, self.observation = self.dynamics.timestep(observation=self.observation, input=action, dt=self.dt)

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
        self.vis.render_state(self.obs_act_collection, self.dt, self.target_altitude)

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