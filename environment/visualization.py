import gymnasium as gym
import subprocess
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import time
import numpy as np
from typing import Deque

# ToDo: make this shit working
class PlotVisualizer():
    def __init__(self):
        pass
       
    def visualize_episode(episode_data: Deque, dt: float, target_altitude: float):
        
        t = [n*dt for n in range(len(episode_data))]

        fig         = plt.figure(figsize=(10, 5))
        ax_alpha    = plt.subplot(3, 3, 1)          # aplha
        ax_alpha.plot(t, episode_data[:,0])
        ax_alpha.set_ylabel('alpha')
        ax_q        = plt.subplot(3, 3, 2)          # q
        ax_q.plot(t, episode_data[:,1])
        ax_q.set_ylabel('q')
        ax_V        = plt.subplot(3, 3, 4)          # V
        ax_V.plot(t, episode_data[:,2])
        ax_theta    = plt.subplot(3, 3, 5)          # theta
        ax_theta.plot(t, episode_data[:,3])
        ax_elevator = plt.subplot(3, 3, 7)          # elevator
        ax_elevator.plot(t, episode_data[:,-2])
        ax_throttle = plt.subplot(3, 3, 8)          # throttle
        ax_throttle.plot(t, episode_data[:,-1])
        ax_xz       = plt.subplot(3, 3, (3,6,9))    # x-z plot
        ax_xz.plot(episode_data[:,4], episode_data[:,5])
        ax_xz.axhline(y=target_altitude, linestyle='--')

        plt.show()
    
    def close(self):
        plt.close()