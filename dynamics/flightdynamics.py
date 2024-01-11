import numpy as np
import scipy.integrate as scint

class Flightdynamics:

    # states:
        # alpha - angle of attack [rad] -> x[0]
        # q     - pitch rate [rad/s]    -> x[1]
        # V     - velocity [m/s]        -> x[2]
        # theta - pitch angle [rad]     -> x[3]
        # x     - x position [m]        -> x[4]
        # z     - z position [m]        -> x[5]
    
    # inputs:
        # elevator - elevator angle [rad] -> u[0]
        # throttle - throttle setting [-]  -> u[1]

    # examplary values for a Diamond DA40
    Zalpha_V0 = -2.3903
    Malpha = -22.5017
    Xalpha = 10.7419
    Mq = -2.7314
    Mv = 0.0216
    Zv_V0 = -0.003251
    Xv = -0.0181
    g = 9.81

    A = np.array([
            [Zalpha_V0, 1., Zv_V0, 0., 0., 0.],
            [Malpha, Mq, Mv, 0., 0., 0.],
            [Xalpha, 0, Xv, -g, 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0, 0.],
            [0., 0., 0., 0., 0., 0.]
        ])
    
    Zeta_V0 = -0.1515
    XdeltaF_V0_sin_alpha0_iF = 0.
    Meta = -20.0759
    MdeltaF = -0.0562
    Xeta = 0.0882
    XdeltaF_cos_alpha0_iF = 1.1737

    B = np.array([
            [Zeta_V0, XdeltaF_V0_sin_alpha0_iF],
            [Meta, MdeltaF],
            [Xeta, XdeltaF_cos_alpha0_iF],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ])
    


    def __init__(self,id,state: np.ndarray):
        '''
        Class describing the flight dynamics of the aircraft
        @param initial_state: np.ndarray of shape (6,1) representing the initial state [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m]].T of the system
        '''
        self.id = id
        self.state = state
        if state.shape != (6,1):
            raise ValueError('Initial state must be of shape (6,1): state = [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m]].T, but is of shape ' + str(state.shape))



    def integrate(self,u,dt):
        '''
        Returns the new state after integration
        @return: new state
        '''
        xdot = self.A @ self.state + self.B @ u
        # Euler-Cauchy Integration (ODE1)
        newState = self.state + dt * xdot
        return newState



    def timestep(self, input: np.ndarray, dt: float) -> np.ndarray:
        '''
        Takes in an input and a timestep and returns the new state of the system
        @param input: np.ndarray of shape (2,1) with [elevator [rad], throttle [N]].T as input
        @param dt: float representing the timestep
        @return: np.ndarray of shape (6,1) representing the new state [alpha [rad], q [rad/s], V [m/s], Theta [rad], x [m], z [m]].T of the system
        '''
        self.state = self.integrate(input,dt)
        return self.state