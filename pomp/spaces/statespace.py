from __future__ import print_function,division
from builtins import range
from six import iteritems

from .configurationspace import *
from .controlspace import *

class VelocitySpace(Dynamics):
    """A velocity differential constraint for a first-order system"""
    def __init__(self,vspace):
        self.vspace = vspace
    def controlSet(self,x):
        return self.vspace
    def derivative(self,x,u):
        return u

class CVSpace(MultiConfigurationSpace):
    """A configuration space containing configuration x velocity"""
    def __init__(self,cspace,vspace):
        MultiConfigurationSpace.__init__(self,cspace,vspace)
        self.cspace = cspace
        self.vspace = vspace
    def interpolator(self,x,y):
        #default uses hermite interpolation
        return HermiteInterpolator(self.split(x),self.split(y))

class CVBoundedSpace(CVSpace):
    """A CVSpace wtih box-bounded velocities"""
    def __init__(self,cspace,vmin,vmax):
        CVSpace.__init__(self,cspace,BoxConfigurationSpace(vmin,vmax))
        
class AccelerationDynamics(Dynamics):
    """An acceleration differential constraint for a second-order system"""
    def __init__(self,aspace):
        self.aspace = aspace
    def controlSet(self,x):
        return self.aspace
    def derivative(self,x,u):
        return list(x[len(x)//2:])+list(u)

class CVControlSpace(KinodynamicSpace):
    """Adapts a configuration/velocity/acceleration space to a ControlSpace.
    State is x = (q,v) and control is u = a
    """
    def __init__(self,cspace,vspace,aspace,dt=0.01,dtmax=0.1):
        KinodynamicSpace.__init__(self,CVSpace(cspace,vspace),AccelerationDynamics(aspace),dt=dt,dtmax=dtmax)
    def trajectory(self,x,u):
        n = len(x)//2
        duration = u[0]
        a = u[1:]
        assert n == len(a)
        path = [x]
        t = 0.0
        while t < duration:
            q,v = path[-1][:n],path[-1][n:]
            dt = min(self.dt,duration-t)
            vnew = vectorops.madd(v,a,dt)
            qnew = vectorops.add(vectorops.madd(q,a,0.5*dt**2),vectorops.mul(v,dt))
            path.append(qnew+vnew)
            t = min(t+self.dt,duration)
        return path
    def nextState(self,x,u):
        n = len(x)//2
        q,v = x[:n],x[n:]
        dt = u[0]
        a = u[1:]
        assert n == len(a)
        vnew = vectorops.madd(v,a,dt)
        qnew = vectorops.add(vectorops.madd(q,a,0.5*dt**2),vectorops.mul(v,dt))
        return qnew+vnew
    def nextState_jacobian(self,x,u):
        n = len(x)//2
        q,v = x[:n],x[n:]
        dt = u[0]
        a = u[1:]
        assert n == len(a)
        Jx = np.eye(len(x))
        Jx[:n,n:] = np.eye(n)*dt
        Ju = np.zeros((len(x),n+1))
        Ju[:n,0] = vectorops.madd(v,a,dt)
        Ju[n:,0] = a
        Ju[:n,1:1+n] = np.eye(n)*(0.5*dt**2)
        Ju[n:,1:1+n] = np.eye(n)*dt
        return Jx,Ju