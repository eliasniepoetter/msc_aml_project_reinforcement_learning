import gymnasium as gym
import subprocess
import matplotlib.pyplot as plt
from flightgear_python.fg_if import FDMConnection
import time

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
                '--timeofday=noon']
        
        # start flightgear and wait until it is ready
        self.flightgear_process = subprocess.Popen(command + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            msg_out = self.flightgear_process.stdout.readline().decode()
            if 'loading cities done' in msg_out:
                time.sleep(5)
                break
            else:
                time.sleep(0.001)
        gym.logger.info('flightgear launched')
        print('flightgear launched')

    def _connect_flightgear(self):
        self.fdm_conn = FDMConnection(fdm_version=24)
        self.fdm_event_pipe = self.fdm_conn.connect_rx('localhost', 5501, self._fdm_callback)
        self.fdm_conn.connect_tx('localhost', 5502)
        self.fdm_conn.start()
        print('flightgear connected')

    def _fdm_callback(self, fdm_data, event_pipe):
        if event_pipe.child_poll():
            alt_m, = event_pipe.child_recv()
            print('commanded height: ' + str(alt_m))
            fdm_data['alt_m'] = alt_m
        return fdm_data
    
    def render_state(self, state):
        # ToDo: implement rendering of state
        self.fdm_event_pipe.parent_send((state,))

    def close(self):
        if self.flightgear_process:
            self.flightgear_process.kill()
            
            