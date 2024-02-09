import gymnasium as gym
import subprocess
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import time
import numpy as np
from typing import Deque

# ToDo: make this shit working
class PlotVisualizer():
    def __init__(self, target_altitude):
        self.target_altitude = target_altitude
        # self._init_plot()
        # self.fig.canvas.draw()
        # plt.show(block=False)

    def reset_plot(self):
        self.t = np.zeros(1)
        self.plot_data = np.zeros((1,8))
    
    def _init_plot(self):
        self.fig, self.axs = plt.subplots(4, 2)
        self.reset_plot()
        ylabels = ['alpha', 'q', 'V', 'Theta', 'x-Pos.', 'z-Pos.', 'elevator', 'throttle']
        for i, ax in enumerate(self.axs.flat):
            ax.set(ylabel=ylabels[i])
            ax.grid(True)
        self.axs[3, 0].set(xlabel='t')
        self.axs[3, 1].set(xlabel='t')
        plt.tight_layout()
        self._plot()

    def _plot(self):
        for i in range(4):
            for j in range(2):
                self.axs[i, j].cla()
                self.axs[i, j].plot(self.t, self.plot_data[:,i*2+j])
                if (i==2) & (j==1):
                    self.axs[i, j].axhline(y=self.target_altitude, linestyle='--')
                
        
    def render_state(self, episode_data: Deque, dt: float, target_altitude: float):
        '''
        Renders the state in a plot
        @param episode_data: Deque of np.ndarrays of shape (9,) representing the state and actions [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m], target_altitude [m], elevator [rad], throttle [0..1]] of the system
        @param dt: float representing the time step [s] of the system
        '''
        self.target_altitude = target_altitude
        self.fig, self.axs = plt.subplots(4, 2)
        plt.rcParams['text.usetex'] = True
        ylabels = [r'angle of attack $\alpha$ [rad]', r'pich rate $q$ [rad]', r'velocity $V$ [m/s]', r'pitch angle $\theta$ [rad]', r'$x$ position [m]', r'$z$ position [m]', r'elevator $\eta$ [rad]', 'throttle $\delta_F$ [-]']
        for i, ax in enumerate(self.axs.flat):
            ax.set(ylabel=ylabels[i])
            ax.grid(True)
        self.axs[3, 0].set(xlabel='time [s]')
        self.axs[3, 1].set(xlabel='time [s]')
        plt.tight_layout()
        
        
        self.t = [n*dt for n in range(len(episode_data))]
        self.plot_data = np.delete(np.array(episode_data), 6, 1) # delete target altitude from plotting data
        self.plot_data[:,2] += 51.4 # add V0 to velocity
        for i in range(4):
            for j in range(2):
                # self.axs[i, j].cla()
                self.axs[i, j].plot(self.t, self.plot_data[:,i*2+j])
                self.axs[i, j].grid(True)
                if (i==2) & (j==1):
                    self.axs[i, j].axhline(y=self.target_altitude, linestyle='--')
                    self.axs[i, j].legend([r'True', r'Target'])
                    self.axs[i, j].grid(True)
        plt.show()

        plt.plot(self.plot_data[:,4],self.plot_data[:,5])
        plt.xlabel(r'$x$ position [m]')
        plt.ylabel(r'$z$ position [m]')
        plt.grid(True)
        
    def close(self):
        plt.close()