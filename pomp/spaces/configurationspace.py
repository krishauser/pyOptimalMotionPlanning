from interpolators import *
from geodesicspace import *
from metric import *
from sets import *
from ..klampt import vectorops
import random       

def sample_hypersphere(d,c,r):
    """Samples a d-dimensional sphere uniformly, centered at c and with
    radius r"""
    assert(d == len(c))
    d = [random.gauss(0,1) for ci in c]
    d = vectorops.unit(d)
    return vectorops.madd(c,d,r)

def sample_hyperball(d,c,r):
    """Samples a d-dimensional ball uniformly, centered at c and with
    radius r"""
    assert(d == len(c))
    rad = pow(random.random(),1.0/d)
    return sample_hypersphere(d,c,rad)

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
        #return sample_hyperball(len(x),x,r)
        res = x[:]
        for i in xrange(len(res)):
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

class CartesianConfigurationSpace(GeodesicSpace,ConfigurationSpace):
    def __init__(self,d):
        self.d = d
    def dimension(self):
        return self.d

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

class MultiConfigurationSpace(MultiSet,GeodesicConfigurationSpace):
    """A cartesian product of multiple ConfigurationSpaces"""
    def __init__(self,*components):
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
    


class ConfigurationSubset(Set):
    """A subset of a ConfigurationSpace"""
    def __init__(self,space):
        self.space = space
    def dimension(self):
        return self.space.dimension()
    def contains(self,x):
        """Returns true if x is in the set"""
        return False
    def sample(self):
        """Sample a point in the set, or return None if not possible"""
        return None
    def project(self,x):
        """If x is not contained in the set, returns a nearby point in the
        set.  Returns None if it is not possible or this function is not
        implemented."""
        return None

class SingletonSubset(ConfigurationSubset):
    """A single point {x}"""
    def __init__(self,space,x):
        self.space = space
        self.x = x
    def contains(self,x):
        return x == self.x
    def sample(self):
        return self.x
    def project(self,x):
        return self.x

class NeighborhoodSubset(ConfigurationSubset):
    """A ball of radius r around a point c"""
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

class SetSubset(ConfigurationSubset):
    """Adapts a XSet in sets.py to the ConfigurationSubset class"""
    def __init__(self,space,subset):
        self.space = space
        self.subset = subset
    def bounds(self):
        return self.subset.bounds()
    def contains(self,x):
        return self.subset.contains(x)
    def sample(self):
        return self.subset.sample()

class BoxSubset(SetSubset):
    """An axis aligned bounding box [bmin,bmax]"""
    def __init__(self,space,bmin,bmax):
        SetSubset.__init__(self,space,BoxSet(bmin,bmax))
    def project(self,x):
        assert len(x)==len(self.subset.bmin)
        assert len(x)==len(self.subset.bmax)
        xnew = x[:]
        for i,(xi,a,b) in enumerate(zip(x,self.subset.bmin,self.subset.bmax)):
            if xi < a:
                xnew[i] = a
            elif xi > b:
                xnew[i] = b
        return xnew

class FiniteSubset(SetSubset):
    """A list of points {x1,...,xn}"""
    def __init__(self,space,points):
        SetSubset.__init__(self,space,FiniteSet(points))
    def project(self,x):
        if len(self.subset.items)==0:
            return None
        (d,pt) = sorted([self.space.distance(x,p) for p in self.subset.items])[0]
        return pt

class MultiSubset(ConfigurationSubset):
    """A cartesian product of subsets."""
    def __init__(self,*components):
        self.components = components
        self.space = MultiConfigurationSpace(*[c.space for c in components])
    def bounds(self):
        cbounds = [c.bounds() for c in self.components]
        if any(c==None for c in cbounds): return None
        bmin,bmax = zip(*cbounds)
        return self.join(bmin),self.join(bmax)
    def contains(self,x):
        for (c,xi) in zip(self.components,self.space.split(x)):
            if not c.contains(xi):
                return False
        return True
    def sample(self):
        return self.space.join(c.sample() for c in self.components)
    def project(self,x):
        try:
            return self.space.join(c.project(xi) for (c,xi) in zip(self.components,self.space.split(x)))
        except TypeError:
            #thrown when project returns None
            return None
