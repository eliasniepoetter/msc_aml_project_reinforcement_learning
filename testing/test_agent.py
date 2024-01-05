from rlenv.environment.agent import classAgent

id = 0
initialState = [1, 1]
testAgent = classAgent(id, initialState)
testAgent.greet()

input = 1
printState = True
testAgent.updateStates(input,printState)