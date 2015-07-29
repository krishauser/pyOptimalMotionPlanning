from controlspace import *
from interpolators import *
from objective import ObjectiveFunction


class CostControlSpace(ControlSpace):
    """Adapts a ControlSpace to a state+cost space.  Appends accumulated
    cost to the state variable."""
    def __init__(self,controlSpace,objective,costmax = float('inf')):
        if not isinstance(objective,ObjectiveFunction):
            raise ValueError("objective argument must be a subclass of ObjectiveFunction")
        self.cspace = MultiConfigurationSpace(controlSpace.configurationSpace(),BoxConfigurationSpace([0.0],[costmax]))
        self.baseSpace = controlSpace
        self.objective = objective
        self.costmax = costmax
    def makeState(self,baseState,cost):
        return baseState+[cost]
    def baseState(self,x):
        return x[:-1]
    def cost(self,x):
        return x[-1]
    def setCostMax(self,costmax):
        if costmax == None:
            costmax = float('inf')
        self.costmax = costmax
        self.cspace.components[1].box.bmax[0] = costmax
    def configurationSpace(self):
        return self.cspace
    def controlSet(self,x):
        return self.baseSpace.controlSet(x[:-1])
    def nextState(self,x,u):
        xbasenext = self.baseSpace.nextState(x[:-1],u)
        cnext = x[-1]+self.objective.incremental(x[:-1],u)
        return xbasenext+[cnext]
    def interpolator(self,x,u):
        cnext = x[-1]+self.objective.incremental(x[:-1],u)
        return MultiInterpolator(self.baseSpace.interpolator(x[:-1],u),LinearInterpolator([x[-1]],[cnext]))

