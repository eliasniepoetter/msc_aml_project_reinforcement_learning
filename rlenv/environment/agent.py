class classAgent:

    # implement longitudinal dynamics below
    f = staticmethod(lambda x,u: [x*u])

    def __init__(self,id,state):
        self.id = id
        self.state = state

    def greet(self):
        print("Hello, I am agent " + str(self.id) + ".")

    def integrate(self,method,dt,input):
        match method:
            case "ode1":
                pass

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
        # self.integrate("ode1",dt,u)
        if printState:
            print("Agent " + str(self.id) + " new state: " + str(self.state))

        

