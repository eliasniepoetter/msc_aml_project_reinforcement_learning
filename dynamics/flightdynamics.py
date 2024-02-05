import numpy as np

class Flightdynamics:

    # states:
        # alpha  - angle of attack [rad]    -> x[0]
        # q      - pitch rate [rad/s]       -> x[1]
        # V      - velocity [m/s]           -> x[2]
        # theta  - pitch angle [rad]        -> x[3]
    
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
            [Zalpha_V0, 1., Zv_V0, 0.],
            [Malpha, Mq, Mv, 0.],
            [Xalpha, 0, Xv, -g],
            [0., 1., 0., 0.],
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
            [0., 0.]
        ])
    


    def __init__(self):
        # initial linearized state
        self.V0 = 51.4
        self.reset()

    def reset(self):
        self.state = np.array([0., 0., 0., 0.])
        
    def integrate(self, u, dt):
        xdot = self.A @ self.state + self.B @ u
        # Euler-Cauchy Integration (ODE1)
        self.state += dt * xdot

    def timestep(self, observation: np.ndarray, input: np.ndarray, dt: float):
        # store old state and integrate state
        old_state = np.copy(self.state)
        self.integrate(input, dt)

        # calculate observation
        gamma = self.state[3] - self.state[0]
        observation[0] = (self.state[2]+self.V0) * np.cos(gamma)    # v_x
        observation[1] = (self.state[2]+self.V0) * np.sin(gamma)    # v_y
        observation[2] = observation[2] + (self.state[2]+self.V0) * np.cos(gamma) * dt   # x_pos
        observation[3] = observation[3] + (self.state[2]+self.V0) * np.sin(gamma) * dt   # z_pos

        # return new state
        return self.state, observation