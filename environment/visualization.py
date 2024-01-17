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
        self._init_plot()
        self.fig.canvas.draw()
        plt.show(block=False)

    def reset_plot(self):
        self.t = np.zeros(1)
        self.plot_data = np.zeros((1,8))
    
    def _init_plot(self):
        self.fig, self.axs = plt.subplots(4, 2)
        self.reset_plot()
        self.lines = []
        self._plot()
        ylabels = ['alpha', 'q', 'V', 'Theta', 'x-Pos.', 'z-Pos.', 'elevator', 'throttle']
        for i, ax in enumerate(self.axs.flat):
            ax.set(ylabel=ylabels[i])
            ax.grid(True)
        self.axs[3, 0].set(xlabel='t')
        self.axs[3, 1].set(xlabel='t')
        plt.tight_layout()

    def _plot(self):
        for i in range(4):
            for j in range(2):
                self.axs[i, j].plot(self.t, self.plot_data[:,i*2+j])
                
        
    def render_state(self, episode_data: Deque, dt: float):
        '''
        Renders the state in a plot
        @param episode_data: Deque of np.ndarrays of shape (8,) representing the state and actions [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m], elevator [rad], throttle [0..1]] of the system
        @param dt: float representing the time step [s] of the system
        '''
        self.t = [n*dt for n in range(len(episode_data))]
        self.plot_data = np.array(episode_data)
        self._plot()
       
        #self.fig.gca().relim()
        #self.fig.gca().autoscale_view()
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        plt.pause(0.000001)

    def close(self):
        plt.close()