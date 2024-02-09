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