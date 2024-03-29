import gymnasium as gym
import subprocess
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import time
import numpy as np
from typing import Deque

class PlotVisualizer():
    def __init__(self):
        pass

    def visualize_episode(self, episode_data: Deque, dt: float, target_altitude: float):
        # time vector
        t = [n*dt for n in range(len(episode_data))]
        # convert deque to numpy array
        episode_data = np.array(episode_data)

        fig         = plt.figure(figsize=(16, 9))
        # alpha
        ax_alpha    = plt.subplot(4, 3, 1)
        ax_alpha.plot(t, episode_data[:,0])
        ax_alpha.set_ylabel('alpha [rad]')
        ax_alpha.grid(True)

        # q
        ax_q        = plt.subplot(4, 3, 2)         
        ax_q.plot(t, episode_data[:,1])
        ax_q.set_ylabel('q [rad/s]')
        ax_q.grid(True)

        # V
        ax_V        = plt.subplot(4, 3, 4)
        ax_V.plot(t, episode_data[:,2]+51.4, label='V')
        ax_V.set_ylabel('V [m/s]')
        ax_V.grid(True)

        # theta
        ax_theta    = plt.subplot(4, 3, 5)          
        ax_theta.plot(t, episode_data[:,3])
        ax_theta.set_ylabel('theta [rad]')
        ax_theta.grid(True)

        # x_pos
        ax_x_pos = plt.subplot(4, 3, 7)          
        ax_x_pos.plot(t, episode_data[:,4])
        ax_x_pos.set_ylabel('x_pos [m]')
        ax_x_pos.grid(True)

        # z_pos
        ax_z_pos = plt.subplot(4, 3, 8)          
        ax_z_pos.plot(t, episode_data[:,5])
        ax_z_pos.axhline(y=target_altitude, linestyle='--')
        ax_z_pos.set_ylabel('z_pos [m]')
        ax_z_pos.grid(True)

        # elevator
        ax_elevator = plt.subplot(4, 3, 10)          
        ax_elevator.plot(t, episode_data[:,-2])
        ax_elevator.set_ylabel('elevator')
        ax_elevator.set_xlabel('t [s]')
        ax_elevator.grid(True)

        # throttle
        ax_throttle = plt.subplot(4, 3, 11)          
        ax_throttle.plot(t, episode_data[:,-1])
        ax_throttle.set_ylabel('throttle')
        ax_throttle.set_xlabel('t [s]')
        ax_throttle.grid(True)

        # x-z plot
        ax_xz       = plt.subplot(4, 3, (3,12))    
        ax_xz.plot(episode_data[:,4], episode_data[:,5])
        ax_xz.axhline(y=target_altitude, linestyle='--')
        ax_xz.set_xlabel('x [m]')
        ax_xz.set_ylabel('z [m]')
        ax_xz.grid(True)
        ax_xz.set_aspect('equal')

        plt.tight_layout()
        plt.show()
        
    def close(self):
        plt.close()