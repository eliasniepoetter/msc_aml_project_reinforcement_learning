import gymnasium as gym
import subprocess
import matplotlib.pyplot as plt
from flightgear_python.fg_if import FDMConnection
import time
import numpy as np

# ToDO: implement visualizer
class PlotVisualizer():
    def __init__(self):
        pass

class FlightGearVisualizer():
    def __init__(self):
        self._launch_flightgear()
        self._connect_flightgear()
        

    def _launch_flightgear(self):
        gym.logger.info('launching flightgear')
        # create the command and arguments to launch flightgear
        # ToDo: adapt path to your system
        ROOT_DIR = r'D:\Program Files\FlightGear 2020.3'
        command = [ROOT_DIR + r'\bin\fgfs.exe',
                   '--fg-root=' + ROOT_DIR + r'\data']
        args = ['--fdm=null', 
                '--max-fps=30', 
                '--native-fdm=socket,out,30,localhost,5501,udp', 
                '--native-fdm=socket,in,30,localhost,5502,udp',
                '--disable-ai-traffic',
                '--disable-real-weather-fetch',
                '--timeofday=noon',
                '--prop:/engines/engine[0]/running=true',
                '--geometry=1920x1080',
                '--airport=EDDS'
                '--prop:/on']
        
        # start flightgear and wait until it is ready
        self.flightgear_process = subprocess.Popen(command + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            msg_out = self.flightgear_process.stdout.readline().decode()
            if 'loading cities done' in msg_out:
                time.sleep(5)
                break
            else:
                time.sleep(0.001)
        gym.logger.info('flightgear ready')

    def _connect_flightgear(self):
        gym.logger.info('connecting to flightgear')
        self.fdm_conn = FDMConnection(fdm_version=24)
        self.fdm_event_pipe = self.fdm_conn.connect_rx('localhost', 5501, self._fdm_callback)
        self.fdm_conn.connect_tx('localhost', 5502)
        self.fdm_conn.start()
        gym.logger.info('flightgear connected')

    def _fdm_callback(self, fdm_data, event_pipe):
        if event_pipe.child_poll():
            state = event_pipe.child_recv()
            fdm_data.alpha_rad = state[0]
            fdm_data.thetadot_rad_per_s = state[1]
            fdm_data.vcas = state[2]
            fdm_data.theta_rad = state[3]
            x_pos = state[4]
            # ToDo: convert x_pos to lat_rad, long_rad            
            #fdm_data.lat_rad = 
            #fdm_data.lon_rad = 
            fdm_data.alt_m = state[5]
        return fdm_data
    
    def render_state(self, state: np.ndarray):
        '''
        Renders the state in flightgear
        @param state: np.ndarray of shape (6,1) representing the state [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m]].T of the system
        '''
        # ToDo: implement rendering of state
        self.fdm_event_pipe.parent_send(state)

    def close(self):
        if self.flightgear_process:
            self.flightgear_process.kill()
            
            