from .configurationspace import *
from .timespace import *
from .interpolators import *
from .sets import *
from .biassets import TimeBiasSet
import differences
import numpy as np

MAX_INTEGRATION_STEPS = 10000

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
        
    def nextState_jacobian(self,x,u):
        """If you want to use numerical optimization methods, implement
        this. It should return a pair of matrices dx_n/dx, dx_n/du.
        Subclasses can use nextState_jacobian_diff to approximate
        the Jacobian."""
        return self.nextState_jacobian_diff(x,u)
    
    def nextState_jacobian_diff(self,x,u,h=1e-4):
        Jx = differences.jacobian_forward_difference((lambda y:self.nextState(y,u)),x,h)
        Ju = differences.jacobian_forward_difference((lambda v:self.nextState(x,v)),u,h)
        return (Jx,Ju)

    def checkDerivatives(self,x,u,baseTol=1e-3):
        Jx,Ju = self.nextState_jacobian(x,u)
        dJx,dJu = self.nextState_jacobian_diff(x,u)
        xtol = max(1,np.linalg.norm(dJx))*baseTol
        utol = max(1,np.linalg.norm(dJu))*baseTol
        res = True
        if np.linalg.norm(Jx-dJx) > xtol:
            print("Derivatives of ControlSpace",self.__class__.__name__,"incorrect in Jacobian x by",np.linalg.norm(Jx-dJx),"diff norm",np.linalg.norm(dJx))
            #print("  Computed",Jx)
            #print("  Differenced",dJx)
            res = False
        if np.linalg.norm(Ju-dJu) > utol:
            print("Derivatives of ControlSpace",self.__class__.__name__,"incorrect in Jacobian u by",np.linalg.norm(Ju-dJu),"diff norm",np.linalg.norm(dJu))
            #print("  Computed",Ju)
            #print("  Differenced",dJu)
            res = False
        return res


class ControlSpaceAdaptor(ControlSpace):
    """Adapts a plain ConfigurationSpace to a ControlSpace. The
    control is simply the next state, and the function f(x,u) simply
    produces the next state x'=u.
    """
    def __init__(self,cspace,nextStateSamplingRange=1.0):
        self.cspace = cspace
        self.nextStateSamplingRange = 1.0
    def configurationSpace(self):
        return self.cspace
    def controlSet(self,x):
        return NeighborhoodSubset(self.cspace,x,self.nextStateSamplingRange)
    def nextState(self,x,u):
        return u.copy()
    def interpolator(self,x,u):
        return self.cspace.interpolator(x,u)
    def connection(self,x,y):
        return [y]
    def nextState_jacobian(self,x,u):
        return np.zeros((len(x),len(x))),np.eye(len(u))


class LTIControlSpace(ControlSpace):
    """Implements a discrete-time, linear time invariant control
    space f(x,u) = Ax+Bu.
    """
    def __init__(self,cspace,controlSet,A,B):
        self.cspace = cspace
        self.uspace = controlSet
        self.A = A
        self.B = B
    def configurationSpace(self):
        return self.cspace
    def controlSet(self,x):
        return self.uspace
    def nextState(self,x,u):
        return (self.A.dot(x) + self.B.dot(u)).tolist()
    def interpolator(self,x,u):
        return self.cspace.interpolator(x,self.nextState(x,u))
    def connection(self,x,y):
        #TODO: solve for multi-step control (if controllable)
        if self.B.shape[1] < self.B.shape[0]: return None
        xn = self.A.dot(x)
        dx = np.asarray(y)-np.asarray(xn)
        Binv = np.linalg.pinv(self.B)
        return [Binv.dot(dx)]
    def nextState_jacobian(self,x,u):
        return self.A,self.B



class Dynamics:
    """A differential equation relating state x to control variable u.
    The derivative() method gives diffential constraints x'=f(x,u)"""
    def controlSet(self,x):
        raise NotImplementedError()
    def derivative(self,x,u):
        raise NotImplementedError()


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
        assert self.dt > 0
        dt0 = self.dt
        if duration / self.dt > MAX_INTEGRATION_STEPS:
            print("Warning, more than",MAX_INTEGRATION_STEPS,"steps requested for KinodynamicSpace",self.__class__.__name__)
            dt0 = duration/MAX_INTEGRATION_STEPS
        while t < duration:
            dx = self.dspace.derivative(path[-1],ub)
            assert len(dx)==len(x),"Derivative %s dimension not equal to state dimension: %d != %d"%(self.dspace.__class__.__name__,len(dx),len(x))
            dt = min(dt0,duration-t)
            xnew = vectorops.madd(path[-1],dx,dt)
            path.append(xnew)
            t = min(t+dt0,duration)
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
        dt0 = self.dt
        if duration / self.dt > MAX_INTEGRATION_STEPS:
            print("Warning, more than",MAX_INTEGRATION_STEPS,"steps requested for KinodynamicSpace",self.__class__.__name__)
            dt0 = duration/MAX_INTEGRATION_STEPS
        t = 0.0
        while t < duration:
            dx = self.dspace.derivative(path[-1][1:],ub)
            assert len(dx)==len(x),"Derivative dimension not equal to state dimension"
            dt = min(dt0,duration-t)
            xnew = vectorops.madd(path[-1][1:],dx,dt)
            path.append([path[-1][0]+dt]+xnew)
            t = min(t+dt0,duration)
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
        dt0 = self.dt
        if duration / self.dt > MAX_INTEGRATION_STEPS:
            print("Warning, more than",MAX_INTEGRATION_STEPS,"steps requested for",self.__class__.__name__)
            dt0 = duration/MAX_INTEGRATION_STEPS
        t = 0.0
        while t < duration:
            dx = self.f(path[-1],ub)
            assert len(dx)==len(x),"Derivative dimension not equal to state dimension"
            dt = min(dt0,duration-t)
            xnew = vectorops.madd(path[-1],dx,dt)
            path.append(xnew)
            t = min(t+dt0,duration)
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
        for start in range(0,len(u),d):
            x = self.base.nextState(x,u[start:start+d])
        return x
    def interpolator(self,x,u):
        d = len(u)/self.n
        res = PathInterpolator([])
        for start in range(0,len(u),d):
            res.edges.append(self.base.interpolator(x,u[start:start+d]))
            x = res.edges[-1].end()
        return res
