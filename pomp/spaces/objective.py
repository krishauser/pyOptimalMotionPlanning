from ..klampt import vectorops

class ObjectiveFunction:
    """Objective function base class.  Measures the cost of a trajectory
    [x0,...,xn],[u1,...,un] or a path [x0,...,xn]"""
    def incremental(self,x,u=None):
        return 0.0
    def terminal(self,x):
        return 0.0
    def cost(self,xpath,upath):
        if upath==None:
            for i in xrange(len(xpath)-1):
                c += self.incremental(xpath[i],None)
            return c+self.terminal(xpath[-1])
        assert len(xpath)==len(upath)+1
        c = 0.0
        for i in xrange(len(upath)):
            c += self.incremental(xpath[i],upath[i])
        return c + self.terminal(xpath[-1])

class PathLengthObjectiveFunction(ObjectiveFunction):
    """Meant for an ControlSpaceAdaptor class"""
    def incremental(self,x,u):
        return vectorops.distance(x,u)

class TimeObjectiveFunction(ObjectiveFunction):
    """Meant for a KinodynamicSpace class"""
    def incremental(self,x,u):
        return u[0]

class LambdaObjectiveFunction(ObjectiveFunction):
    """Adapts a function incremental(x,u) and/or terminal(x) to an
    ObjectiveFunction object.
    """
    def __init__(self,incremental=None,terminal=None):
        self.fincremental = incremental
        self.fterminal = terminal
    def incremental(self,x,u=None):
        if self.fincremental: return self.fincremental(x,u)
        else: return 0.0
    def terminal(self,x):
        if self.fterminal: return self.fterminal(x)
        else: return 0.0

class PointwiseObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self,space,pointwise,timestep=0.01):
        self.space = space
        self.fpointwise = pointwise
        self.timestep = timestep
    def incremental(self,x,u):
        e = self.space.interpolator(x,u)
        l = e.length()
        if l == 0: return 0
        t = 0
        c = 0
        while t < l:
            c += self.fpointwise(e.eval(t / l),u)
            t += self.timestep
        return c*self.timestep
