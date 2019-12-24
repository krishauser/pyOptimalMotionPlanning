from .sets import *
import math
import random

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

class BoxBiasSet(BoxSet):
    """A set that samples a box near its extrema, helpful for bang-bang control.
    
    Assume the box is [-1,1]^d.  A dimension k is picked, and a variable s is 
    sampled by s = rand()^(1/c) where rand() samples uniformly from [0,1].  
    Then its sign is randomly flipped with probability 0.5.  u[k] is then
    set to s. The remaining dimensions are sampled as usual.
    
    To get back to an arbitrary box the range [-1,1]^d is simply scaled.
    """
    def __init__(self,bmin,bmax,concentration=3):
        BoxSet.__init__(self,bmin,bmax)
        if math.isinf(concentration):
            self.power = 0
        else:
            self.power = 1.0/concentration
            
    def sample(self):
        res = BoxSet.sample(self)
        d = random.randint(0,len(res)-1)
        (a,b) = (self.bmin[d],self.bmax[d])
        sign = random.choice([-1,1])
        s = math.pow(random.random(),self.power)
        res[d] = (a+b)*0.5 + sign*(b-a)*0.5*s
        return res


class InfiniteBiasSet(Set):
    """An infinite set of dimension d.
    
    The variable is sampled from a multivariate gaussian distribution with
    the given concentration parameter
    """
    def __init__(self,d,concentration=1.0):
        self.d = d
        self.concentration = concentration
        
    def __str__(self):
        return self.__class__.__name__+" of dim "+str(self.d)
        
    def dimension(self):
        return self.d
        
    def bounds(self):
        return None
        
    def sample(self):
        return (np.random.randn(self.d)/self.concentration).tolist()
        
    def contains(self,x):
        assert len(x) == self.d
        return True
    
    def project(self,x):
        return x

    def signedDistance(self,x):
        return -float('inf')
    
    def signedDistance_gradient(self,x):
        return np.zeros(len(x))

