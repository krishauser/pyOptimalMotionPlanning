from configurationspace import *
from timespace import *
from interpolators import *
from sets import *

class ControlSpace:
    """A space regarding control variable u and its effects on members of a
    ConfigurationSpace.  This class defines the transformation
    xnext = nextState(x,u).
    """
    def __init__(self):
        pass
    def configurationSpace(self):
        """Returns the configuration space associated with this."""
        return None
    def controlSet(self,x):
        """Returns the set from which controls should be sampled"""
        raise NotImplementedError()
    def nextState(self,x,u):
        """Produce the next state after applying control u to state x"""
        raise NotImplementedError()
    def interpolator(self,x,u):
        """Returns the interpolator that goes from x to self.nextState(x,u)"""
        raise NotImplementedError()
    def connection(self,x,y):
        """Returns the sequence of controls that connects x to y, if
        applicable"""
        return None

class ControlSpaceAdaptor(ControlSpace):
    """Adapts a plain ConfigurationSpace to a ControlSpace"""
    def __init__(self,cspace,nextStateSamplingRange=1.0):
        self.cspace = cspace
        self.nextStateSamplingRange = 1.0
    def configurationSpace(self):
        return self.cspace
    def controlSet(self,x):
        return NeighborhoodSubset(self.cspace,x,self.nextStateSamplingRange)
    def nextState(self,x,u):
        return u
    def interpolator(self,x,u):
        return self.cspace.interpolator(x,u)
    def connection(self,x,y):
        return [y]

class Dynamics:
    """A differential equation relating state x to control variable u.
    The derivative() method gives diffential constraints x'=f(x,u)"""
    def controlSet(self,x):
        raise NotImplementedError()
    def derivative(self,x,u):
        raise NotImplementedError()

class TimeBiasSet(BoxSet):
    """A set for an integration time variable that samples more-or-less
    uniformly from the *outcome* state, given a maximum integration duration
    and a space of controls uspace.

    It assumes the next state is obtained by the integral
       int[0,T] f(x(t),u) dt
    and the function f(x,u) is not degenerate.  With this assumption, the
    the volume of the reachable set grows proportionally to T^d where d
    is the control dimension.  Hence, the sampler samples T from the range
    [0,tmax] according to the distribution U(0,1)^(1/d)*tmax.  In practice,
    this places more samples toward the tail end of the integration region. 
    """
    def __init__(self,tmax,uspace):
        BoxSet.__init__(self,[0],[tmax])
        self.tmax = tmax
        self.controlDimension = len(uspace.sample())
        #if self.controlDimension == 1:
        #    self.controlDimension = 2
    def sample(self):
        #plain time sampling
        #return [random.uniform(0,self.tmax)]
        #sampling from upper half
        #return [random.uniform(self.tmax*0.5,self.tmax)]
        #sampling with a power law bias
        return [math.pow(random.random(),1.0/self.controlDimension)*self.tmax]

class KinodynamicSpace (ControlSpace):
    """A control space that adapts a dynamic space to a cspace.

    The control is u=(dt,ubase).  The integration duration dt is sampled from
    the range [0,dtmax].

    The derivative function x'=f(x,ubase) is translated into a nextState
    function xnext = g(x,u) via integration at the resolution
    dt.
    """
    def __init__(self,cspace,dspace,dt=0.01,dtmax=0.1):
        """Provide a ConfigurationSpace and a Dynamics and this class will
        merge them together"""
        self.cspace = cspace
        self.dspace = dspace
        self.dt = dt
        self.dtmax = dtmax
    def configurationSpace(self):
        return self.cspace
    def controlSet(self,x):
        #TEMP: try long extensions
        return MultiSet(TimeBiasSet(self.dtmax,self.dspace.controlSet(x)),self.dspace.controlSet(x))
        #return MultiSet(BoxSet([0],[self.dtmax]),self.dspace.controlSet(x))
    def trajectory(self,x,u):
        duration = u[0]
        ub = u[1:]
        path = [x]
        t = 0.0
        while t < duration:
            dx = self.dspace.derivative(path[-1],ub)
            assert len(dx)==len(x),"Derivative dimension not equal to state dimension"
            dt = min(self.dt,duration-t)
            xnew = vectorops.madd(path[-1],dx,dt)
            path.append(xnew)
            t = min(t+self.dt,duration)
        return path
    def nextState(self,x,u):
        return self.trajectory(x,u)[-1]
    def interpolator(self,x,u):
        res = PiecewiseLinearInterpolator(self.trajectory(x,u))
        if hasattr(self.cspace,'geodesic'):
            res.geodesic = self.cspace.geodesic
        return res
    def connection(self,x,y):
        return None


class TimedKinodynamicSpace (ControlSpace):
    """Just like KinodynamicSpace except the state variable maintains the
    current time in the 0'th entry.

    State is x=(t,xbase), and control is u=(dt,ubase)."""
    def __init__(self,cspace,dspace,dt=0.01,dtmax=0.1):
        """Provide a ConfigurationSpace and a Dynamics and this class will
        merge them together"""
        self.cspace = cspace
        self.tcspace = TCSpace(cspace)
        self.dspace = dspace
        self.dt = dt
        self.dtmax = dtmax
    def configurationSpace(self):
        return self.tcspace
    def controlSet(self,x):
        return MultiSet(TimeBiasSet(self.dtmax,self.dspace.controlSet(x)),self.dspace.controlSet(x))
        #return MultiSet(BoxSet([0],[self.dtmax]),self.dspace.controlSet(x))
    def trajectory(self,x,u):
        duration = u[0]
        ub = u[1:]
        path = [x]
        t = 0.0
        while t < duration:
            dx = self.dspace.derivative(path[-1][1:],ub)
            assert len(dx)==len(x),"Derivative dimension not equal to state dimension"
            dt = min(self.dt,duration-t)
            xnew = vectorops.madd(path[-1][1:],dx,dt)
            path.append([path[-1][0]+dt]+xnew)
            t = min(t+self.dt,duration)
        return path
    def nextState(self,x,u):
        return self.trajectory(x,u)[-1]
    def interpolator(self,x,u):
        res = PiecewiseLinearInterpolator(self.trajectory(x,u))
        if hasattr(self.cspace,'geodesic'):
            res.geodesic = self.cspace.geodesic
        return res
    def connection(self,x,y):
        return None

class LambdaKinodynamicSpace (ControlSpace):
    """A control space that takes a cspace, a state-invariant control space,
    and a dynamics function, and produces nextState by integration.

    If you need a state-varying control space, use KinodynamicSpace.

    Because it needs a duration of integration, the control is u=(dt,ubase)

    The dynamics function produces x'=f(x,ubase). It is translated into a
    nextState function xnext = g(x,u) via Euler integration at the resolution
    dt.
    """
    def __init__(self,cspace,uspace,f,dt=0.01,dtmax=0.1):
        """Provide a ConfigurationSpace, a Set, and a function x'=f(x,u),
        and this class will merge them together into a ControlSpace."""
        self.cspace = cspace
        self.uspace = uspace
        self.f = f
        self.dt = dt
        self.dtmax = dtmax
    def configurationSpace(self):
        return self.cspace
    def controlSet(self,x):
        return MultiSet(TimeBiasSet(self.dtmax,self.uspace),self.uspace)
        #return MultiSet(BoxSet([0],[self.dtmax]),self.uspace)
    def trajectory(self,x,u):
        duration = u[0]
        ub = u[1:]
        path = [x]
        t = 0.0
        while t < duration:
            dx = self.f(path[-1],ub)
            assert len(dx)==len(x),"Derivative dimension not equal to state dimension"
            dt = min(self.dt,duration-t)
            xnew = vectorops.madd(path[-1],dx,dt)
            path.append(xnew)
            t = min(t+self.dt,duration)
        return path
    def nextState(self,x,u):
        return self.trajectory(x,u)[-1]
    def interpolator(self,x,u):
        res = PiecewiseLinearInterpolator(self.trajectory(x,u))
        if hasattr(self.cspace,'geodesic'):
            res.geodesic = self.cspace.geodesic
        return res
    def connection(self,x,y):
        return None


class RepeatedControlSpace(ControlSpace):
    """A control space that selects several control variables u1,...,un and
    propagates them incrementally.
    nextState(...(nextState(nextState(x,u1),u2)...,un).
    """
    def __init__(self,basespace,n):
        self.base = basespace
        self.n = n
        assert n >= 1
    def configurationSpace(self):
        return self.base.configurationSpace()
    def controlSet(self,x):
        #TODO: configuration dependent base space control sets
        return MultiSet(*[self.base.controlSet(x)]*self.n)
    def nextState(self,x,u):
        d = len(u)/self.n
        for start in xrange(0,len(u),d):
            x = self.base.nextState(x,u[start:start+d])
        return x
    def interpolator(self,x,u):
        d = len(u)/self.n
        res = PathInterpolator([])
        for start in xrange(0,len(u),d):
            res.edges.append(self.base.interpolator(x,u[start:start+d]))
            x = res.edges[-1].end()
        return res
