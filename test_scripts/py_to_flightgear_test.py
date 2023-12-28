"""
Start FlightGear with `--fdm=null --max-fps=30 --native-fdm=socket,out,30,localhost,5501,udp --native-fdm=socket,in,30,localhost,5502,udp`
(you probably also want `--fdm=null` and `--max-fps=30` to stop the simulation fighting with
these external commands)

"""
import time
import math
from flightgear_python.fg_if import FDMConnection, CtrlsConnection


def fdm_callback(fdm_data, event_pipe):
    fdm_data.alt_m += 1
    fdm_data.lat_rad += 1e-6 
    fdm_data.theta_rad += 0.01
    return fdm_data

if __name__ == '__main__':
    # connect fligth dynamics model
    fdm_conn = FDMConnection(fdm_version=24)  # May need to change version from 24
    fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    fdm_conn.connect_tx('localhost', 5502)
    fdm_conn.start()  # Start the FDM RX/TX loop

    while True:
        time.sleep(0.5)