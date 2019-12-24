from __future__ import print_function,division
from builtins import range
from six import iteritems

from .interpolators import *
from .geodesicspace import *
from .metric import *
from .sets import *
from ..klampt import vectorops
from . import differences
import random
import numpy as np


class ConfigurationSpace(Set):
    """A base class for a configuration space. At a minimum, subclasses
    should override the sample() and feasible() methods."""
    def __init__(self):
        pass
    
    def dimension(self):
        """Returns the number of entries of a configuration"""
        try:
            x = self.sample()
            return len(x)
        except NotImplementedError:
            raise NotImplementedError()
        
    def intrinsicDimension(self):
        """Returns the number of true degrees of freedom, which may be
        less than dimension() if the representation is redundant."""
        return self.dimension()
    
    def sample(self):
        """Sample a random configuration from the space"""
        raise NotImplementedError("ConfigurationSpace is unbounded")
    
    def sampleNeighborhood(self,x,r):
        """Default samples each element from [xi-r,xi+r]"""
        #return sample.sample_hyperball(len(x),x,r)
        res = x[:]
        for i in range(len(res)):
            res[i] += random.uniform(-r,r)
        return res
    
    def feasible(self,x):
        """Return true if the configuration is feasible"""
        return True

    def contains(self,x):
        return self.feasible(x)
    
    def distance(self,a,b):
        """A distance metric. Default uses euclidean distance"""
        return euclideanMetric(a,b)
    
    def interpolator(self,x,y):
        """Default uses straight line path"""
        return LinearInterpolator(x,y)
    
    def clearance(self,x):
        """Returns a signed distance from x to obstacles.  Should be
        > 0 if x is outside obstacles (feasible(x)=True), < 0 if x
        is inside (feasible(x)=False), and 0 at the boundary.
        
        This does NOT have to be the exact clearance value to be
        useful.
        
        The return value can also be a list of distances to separate
        obstacles, such that the point is feasible if the minimum
        of the vector is > 0.
        
        This should be implemented if you are using numerical
        optimization methods.
        """
        raise NotImplementedError()
    
    def clearance_gradient(self,x):
        """Returns the gradient of the distance-to-obstacle function @ x.
        
        If clearance(x) returns a vector, then this should be a matrix.
        """
        return differences.jacobian_finite_difference(self.clearance,x,1e-4)


class CartesianConfigurationSpace(GeodesicSpace,ConfigurationSpace):
    def __init__(self,d):
        self.d = d
        
    def __str__(self):
        return "Cartesian C-Space R^"+str(d)

    def dimension(self):
        return self.d
    
    def clearance(self,x):
        return float('inf')
    
    def clearance_gradient(self,x):
        return np.zeros(len(x))


class GeodesicConfigurationSpace(GeodesicSpace,ConfigurationSpace):
    """A configuration space with a custom geodesic"""
    def __init__(self,geodesic):
        self.geodesic = geodesic

    def dimension(self):
        return self.geodesic.dimension()

    def distance(self,a,b):
        return self.geodesic.distance(a,b)

    def interpolate(self,a,b,u):
        return self.geodesic.interpolate(a,b,u)

    def interpolator(self,x,y):
        return GeodesicInterpolator(x,y,geodesic)

    def clearance(self,x):
        return float('inf')
    
    def clearance_gradient(self,x):
        return np.zeros(len(x))


class BoxConfigurationSpace(GeodesicSpace,ConfigurationSpace):
    """A subset of cartesian space in vector bounds [bmin,bmax].
    fills out the dimension, sample, and feasible methods"""
    def __init__(self,bmin,bmax):
        self.box = BoxSet(bmin,bmax)

    def dimension(self):
        return self.box.dimension()

    def bounds(self):
        return self.box.bounds()

    def sample(self):
        return self.box.sample()

    def feasible(self,x):
        return self.box.contains(x)
    
    def project(self,x):
        return self.box.project(x)

    def clearance(self,x):
        assert len(x) == len(self.box.bmin)
        res = []
        for (xi,a,b) in zip(x,self.box.bmin,self.box.bmax):
            res.append(xi-a)
            res.append(b-xi)
        return res
    
    def clearance_gradient(self,x):
        res = np.zeros((len(x)*2,len(x)))
        assert len(x) == len(self.box.bmin)
        for i,(xi,a,b) in enumerate(zip(x,self.box.bmin,self.box.bmax)):
            res[i*2,i] = 1.0
            res[i*2+1,i] = -1.0
        return res


class MultiConfigurationSpace(MultiSet,GeodesicConfigurationSpace):
    """A cartesian product of multiple ConfigurationSpaces"""
    def __init__(self,*components):
        for c in components:
            if not isinstance(c,ConfigurationSpace):
                raise ValueError("Need to provide ConfigurationSpace objects to MultiConfigurationSpace")
        MultiSet.__init__(self,*components)
        geodesics = []
        for c in self.components:
            if hasattr(c,'geodesic'):
                geodesics.append(c.geodesic)
            elif isinstance(c,GeodesicSpace):
                geodesics.append(c)
            else:
                geodesics.append(CartesianSpace(c.dimension()))
        self.geodesic = MultiGeodesicSpace(*geodesics)

    def __str__(self):
        return "MultiConfigurationSpace("+','.join(str(s) for s in self.components)+")"

    def setDistanceWeights(self,weights):
        assert len(weights) == len(self.geodesic.componentWeights)
        self.geodesic.componentWeights = weights

    def intrinsicDimension(self):
        return sum(c.intrinsicDimension() for c in self.components)

    def sampleNeighborhood(self,x,r):
        #TODO: take weights into account
        return self.join(c.sampleNeighborhood(xi,r) for xi,c in zip(self.split(x),self.components))

    def feasible(self,x):
        for xi,c in zip(self.split(x),self.components):
            if not c.feasible(xi): return False
        return True

    def distance(self,a,b):
        return self.geodesic.distance(a,b)

    def interpolator(self,x,y):
        return MultiInterpolator(*[c.interpolator(ai,bi) for ai,bi,c in zip(self.split(x),self.split(y),self.components)],weights=self.geodesic.componentWeights)
    
    def clearance(self,x):
        return np.hstack([c.clearance(xi) for (xi,c) in zip(self.split(x),self.components)])
    
    def clearance_gradient(self,x):
        xs = self.split(x)
        grads = [c.clearance_gradient(xi) for (xi,c) in zip(xs,self.components)]
        #build a block-diagonal matrix
        ng = 0
        for g in grads:
            if len(g.shape) == 2:
                ng += g.shape[0]
            else:
                ng += 1
        grad = np.zeros((ng,len(x)))
        i = 0
        j = 0
        for g,xi in zip(grads,xs):
            assert g.shape[-1] == len(xi)
            if len(g.shape) == 2:
                grad[i:i+g.shape[0],j:j+g.shape[1]] = g
                i += g.shape[0]
            else:
                grad[i,j:j+g.shape[0]] = g
                i += 1
            j += g.shape[-1]
        return grad

class SingletonSubset(Set):
    """A single point, with a distance given by a ConfigurationSpace.
    """
    def __init__(self,space,c):
        self.space = space
        self.c = c

    def contains(self,x):
        return x == self.c

    def sample(self):
        return self.c

    def project(self,x):
        return self.c

    def signedDistance(self,x):
        return self.space.distance(x,self.c)


class NeighborhoodSubset(Set):
    """A ball of radius r around a point c, with a distance given by
    a ConfigurationSpace.
    """
    def __init__(self,space,c,r):
        self.space = space
        self.c = c
        self.r = r

    def contains(self,x):
        return self.space.distance(x,self.c) <= self.r

    def sample(self):
        return self.space.sampleNeighborhood(self.c,self.r)

    def project(self,x):
        d = self.space.distance(x,self.c)
        if d <= self.r: return x
        return self.space.interpolate(x,self.c,(d-self.r)/d)

    def signedDistance(self,x):
        d = self.space.distance(x,self.c)
        return d - self.r


class FiniteSubset(FiniteSet):
    """A set of points {x1,...,xn} in a configuration space"""
    def __init__(self,space,points):
        self.space = space
        FiniteSet.__init__(self,points,metric=space.distance)

