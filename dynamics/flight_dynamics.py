import numpy as np

class Dynamics:

    # states:
        # alpha - angle of attack [rad] -> x[0]
        # q     - pitch rate [rad/s]    -> x[1]
        # V     - velocity [m/s]        -> x[2]
        # theta - pitch angle [rad]     -> x[3]


    def __init__(self,id,state):
        self.id = id
        self.state = state


    def greet(self):
        print("Hello, I am agent " + str(self.id) + ".")


    def dynamics(self,state,input):
        x = state
        u = input

        A = np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.]
        ])

        
    def integrate(self,method,dt,input):
        match method:
            case "ode1":
                self.state = self.dynamics(self.state,input)

            case "ode2":
                pass

            case "ode4":
                pass

            case _:
                print("Error: integration method not recognized")


    def updateStates(self,input,printState):
        u = input
        if printState:
            print("Agent " + str(self.id) + " old state: " + str(self.state))

        dt = 0.1
        self.integrate("ode1",dt,u)
        if printState:
            print("Agent " + str(self.id) + " new state: " + str(self.state))

    


        

