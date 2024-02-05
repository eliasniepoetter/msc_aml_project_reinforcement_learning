import matplotlib.pyplot as plt
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
        ax_alpha    = plt.subplot(3, 3, 1)
        ax_alpha.plot(t, episode_data[:,0])
        ax_alpha.set_ylabel('alpha [rad]')

        # q
        ax_q        = plt.subplot(3, 3, 2)         
        ax_q.plot(t, episode_data[:,1])
        ax_q.set_ylabel('q [rad/s]')
        
        # V
        ax_V        = plt.subplot(3, 3, 4)
        ax_V.plot(t, episode_data[:,2]+51.4, lable='V')
        ax_V.plot(t, episode_data[:,4], label='v_x', linestyle='-.')
        ax_V.plot(t, episode_data[:,5], label='v_z', linestyle='--')
        ax_V.set_ylabel('V [m/s]')
        ax_V.legend()

        # theta
        ax_theta    = plt.subplot(3, 3, 5)          
        ax_theta.plot(t, episode_data[:,3])
        ax_theta.set_ylabel('theta [rad]')

        # elevator
        ax_elevator = plt.subplot(3, 3, 7)          
        ax_elevator.plot(t, episode_data[:,-2])
        ax_elevator.set_ylabel('elevator')
        ax_elevator.set_xlabel('t [s]')
        
        # throttle
        ax_throttle = plt.subplot(3, 3, 8)          
        ax_throttle.plot(t, episode_data[:,-1])
        ax_throttle.set_ylabel('throttle')
        ax_throttle.set_xlabel('t [s]')

        # x-z plot
        ax_xz       = plt.subplot(3, 3, (3,9))    
        ax_xz.plot(episode_data[:,6], episode_data[:,7])
        ax_xz.axhline(y=target_altitude, linestyle='--')
        ax_xz.set_xlabel('x [m]')
        ax_xz.set_ylabel('z [m]')

        plt.show()
    
    def close(self):
        plt.close()