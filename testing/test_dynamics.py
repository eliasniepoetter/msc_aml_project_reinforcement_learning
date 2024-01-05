from dynamics.flight_dynamics import Dynamics

id = 0
initialState = [0, 0, 0, 0]
testAgent = Dynamics(id, initialState)
testAgent.greet()

input = 1
printState = True
testAgent.updateStates(input,printState)