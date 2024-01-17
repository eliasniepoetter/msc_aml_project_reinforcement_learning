import gymnasium as gym
import subprocess
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
from flightgear_python.fg_if import FDMConnection
import time
import numpy as np

# ToDo: make this shit working
class PlotVisualizer():
    def __init__(self):
        self._init_plot()
        #self.ani = FuncAnimation(self.fig, self.render_state, blit=True)
        self.fig.canvas.draw()
        self.axbackground = [self.fig.canvas.copy_from_bbox(ax.bbox) for axs in self.axs for ax in axs]
        plt.show(block=False)

    def reset_plot(self):
        self.t = []
        #self.alpha = []
        #self.q = []
        #self.V = []
        #self.Theta = []
        #self.x = []
        #self.z = []
        #self.elevator = []
        #self.throttle = []
        self.plot_data = [[],[],[],[],[],[],[],[]]
    
    def _init_plot(self):
        self.fig, self.axs = plt.subplots(4, 2)
        self.reset_plot()
        self.lines = []
        for i in range(int(len(self.plot_data)/2)):
            for j in range(2):
                temp_lines, = self.axs[i, j].plot(self.t, self.plot_data[i*2+j])
                self.lines.append(temp_lines)
        ylabels = ['alpha', 'q', 'V', 'Theta', 'x-Pos.', 'z-Pos.', 'elevator', 'throttle']
        for i, ax in enumerate(self.axs.flat):
            ax.set(ylabel=ylabels[i])
            ax.set_xlim(0, 100)
            ax.grid(True)
        self.axs[3, 0].set(xlabel='t')
        self.axs[3, 1].set(xlabel='t')
        plt.tight_layout()

    def _plot(self):
        #self.axs[0, 0].plot(self.t, self.alpha)
        #self.axs[0, 1].plot(self.t, self.q)
        #self.axs[1, 0].plot(self.t, self.V)
        #self.axs[1, 1].plot(self.t, self.Theta)
        #self.axs[2, 0].plot(self.t, self.x)
        #self.axs[2, 1].plot(self.t, self.z)
        #self.axs[3, 0].plot(self.t, self.elevator)
        #self.axs[3, 1].plot(self.t, self.throttle)
        for i in range(len(self.plot_data)):
                self.lines[i].set_data(self.t, self.plot_data[i])
        
    def render_state(self, state: np.ndarray, action: np.ndarray, current_t: float):
        '''
        Renders the state in a plot
        @param state: np.ndarray of shape (6,1) representing the state [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m]].T of the system
        @param action: np.ndarray of shape (2,1) representing the action [elevator [rad], throttle [0..1]].T of the system
        @param dt: float representing the time step [s] of the system
        '''
        self.t.append(current_t)
        #self.alpha.append(state[0])
        #self.q.append(state[1])
        #self.V.append(state[2])
        #self.Theta.append(state[3])
        #self.x.append(state[4])
        #self.z.append(state[5])
        #self.elevator.append(action[0])
        #self.throttle.append(action[1])
        for i, st in enumerate(state):
            self.plot_data[i].append(st)
        for i, ac in enumerate(action):
            self.plot_data[i+len(state)].append(ac)
        self._plot()

        for background in self.axbackground:
            self.fig.canvas.restore_region(background)

        for i in range(int(len(self.plot_data)/2)):
            for j in range(2):
                self.axs[i, j].draw_artist(self.lines[i*2+j])
                self.fig.canvas.blit(self.axs[i, j].bbox)

        
        #self.fig.gca().relim()
        #self.fig.gca().autoscale_view()
        #self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()

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
            
            