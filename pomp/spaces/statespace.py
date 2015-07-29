from configurationspace import *
from controlspace import *

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
        return x[len(x)/2:]+u

class CVControlSpace(KinodynamicSpace):
    """Adapts a configuration/velocity/acceleration space to a ControlSpace.
    State is x = (q,v) and control is u = a
    """
    def __init__(self,cspace,vspace,aspace,dt=0.01,dtmax=0.1):
        KinodynamicSpace.__init__(self,CVSpace(cspace,vspace),AccelerationDynamics(aspace),dt=dt,dtmax=dtmax)
