from __future__ import print_function,division
from builtins import range
from six import iteritems

import math
import random
from . import differences
from . import sampling
from ..klampt import vectorops
import numpy as np


class Set:
    """Abstract base class for a set in a d-dimensional vector space. 
    
    A set, at the minimum, has an inside-outside test.  It may be
    optionally bounded and sample-able.
    """
    def __str__(self):
        return self.__class__.__name__
        
    def dimension(self):
        """Returns the number of entries of an element of this set"""
        try:
            x = self.sample()
            return len(x)
        except NotImplementedError:
            raise NotImplementedError("Set "+str(self)+" does not implement dimension()")
            
    def bounds(self):
        """Returns a pair (bmin,bmax) giving an axis-aligned bounding box
        on items in this set.  Return None if no bound can be determined."""
        raise NotImplementedError("Set "+str(self)+" does not implement bounds()")
        
    def sample(self):
        """Sample a random value from the set"""
        raise NotImplementedError("Set "+str(self)+" does not implement sample()")
        
    def contains(self,x):
        """Returns True if x is in the set."""
        raise NotImplementedError("Set "+str(self)+" does not implement contains()")
    
    def project(self,x):
        """If x is not contained in the set, returns a nearby point in the
        set.  If is is contained, just returns x."""
        raise NotImplementedError("Set "+str(self)+" does not implement project()")

    def signedDistance(self,x):
        """Returns a signed distance function d(x) that is > 0 when x is
        outside of the set, and < 0 when x is inside.  d(x)=0 if x is on
        the boundary.
        
        Required for numerical optimization methods."""
        raise NotImplementedError("Set "+str(self)+" does not implement signedDistance()")
    
    def signedDistance_gradient(self,x):
        """Required for numerical optimization methods"""
        return differences.gradient_forward_difference(self.signedDistance,x,1e-4)


class SingletonSet(Set):
    """A single point {x}"""
    def __init__(self,x):
        self.x = x

    def bounds(self):
        return [self.x,self.x]

    def contains(self,x):
        return x == self.x

    def sample(self):
        return self.x

    def project(self,x):
        return self.x

    def signedDistance(self,x):
        d = np.asarray(x)-np.asarray(self.x)
        return np.dot(d,d)
    
    def signedDistance_gradient(self,x):
        d = np.asarray(x)-np.asarray(self.x)
        return 2*d


class NeighborhoodSet(Set):
    """A ball of radius r around a point c"""
    def __init__(self,c,r):
        self.c = c
        self.r = r
        
    def bounds(self):
        return [[v-self.r for v in self.c],[v+self.r for v in self.c]]

    def contains(self,x):
        return vectorops.distance(x,self.c) <= self.r

    def sample(self):
        return sampling.sample_hyperball(len(self.c),self.c,self.r)

    def project(self,x):
        d = vectorops.distance(x,self.c)
        if d <= self.r: return x
        return vectorops.madd(x,vectorops.sub(self.c,x),(d-self.r)/d)

    def signedDistance(self,x):
        d = vectorops.distance(x,self.c)
        return d - self.r




class FiniteSet(Set):
    """Represents a finite set of objects in a vector space."""
    def __init__(self,items,metric=None):
        self.items = items
        if metric is None:
            self.metric = vectorops.distance
        else:
            self.metric = metric
    def bounds(self):
        bmin = self.items[0][:]
        bmax = self.items[0][:]
        for v in self.items[1:]:
            for i in range(len(v)):
                if v[i] < bmin[i]: bmin[i] = v[i]
                elif v[i] > bmax[i]: bmax[i] = v[i]
        return (bmin,bmax)
    def sample(self):
        return random.choice(self.items)
    def contains(self,x):
        return x in self.items
    def project(self,x):
        if len(self.items)==0:
            return None
        (d,pt) = sorted([(self.metric(x,p),p) for p in self.items])[0]
        return pt
    def signedDistance(self,x):
        if self.metric is not vectorops.distance:
            print("WARNING: FiniteSet metric is not euclidean distance, treating as Euclidean in signedDistance")
        mind = float('inf')
        x = np.asarray(x)
        for v in self.items:
            v = np.asarray(v)
            d = self.metric(x,v)
            mind = min(d,mind)
        return mind
    def signedDistance_gradient(self,x):
        if self.metric is not vectorops.distance:
            print("WARNING: FiniteSet metric is not euclidean distance, treating as Euclidean in signedDistance")
        mind = float('inf')
        minv = None
        x = np.asarray(x)
        for v in self.items:
            v = np.asarray(v)
            d = np.dot(x-v,x-v)
            if d < mind:
                minv = v
            mind = min(d,mind)
        if minv is None:
            return np.zeros(len(x))
        return 2*(x-minv)


class BoxSet(Set):
    """Represents an axis-aligned box in a vector space."""
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin,self.bmax)
    def sample(self):
        return [random.uniform(a,b) for (a,b) in zip(self.bmin,self.bmax)]
    def contains(self,x):
        assert len(x)==len(self.bmin)
        for (xi,a,b) in zip(x,self.bmin,self.bmax):
            if xi < a or xi > b:
                return False
        return True
    def project(self,x):
        assert len(x)==len(self.bmin)
        assert len(x)==len(self.bmax)
        xnew = x[:]
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            if xi < a:
                xnew[i] = a
            elif xi > b:
                xnew[i] = b
        return xnew
    def signedDistance(self,x):
        xclamp = np.zeros(len(x))
        assert len(x)==len(self.bmin)
        mindist = float('inf')
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            xclamp[i] = min(b,max(xi,a))
            mindist = min(mindist,xi-a,b-xi)
        if mindist < 0:
            #outside
            x = np.asarray(x)
            return np.dot(x-xclamp,x-xclamp)
        else:
            #inside
            return -mindist
    def signedDistance_gradient(self,x):
        xclamp = np.empty(len(x))
        assert len(x)==len(self.bmin)
        mindist = float('inf')
        imindist = None
        iminsign = 1.0
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            xclamp[i] = min(b,max(xi,a))
            if xi-a < mindist:
                imindist = i
                iminsign = -1.0
                mindist = xi-a
            if b-xi < mindist:
                imindist = i
                iminsign = 1.0
                mindist = b-xi
        if mindist < 0:
            #outside
            x = np.asarray(x)
            return 2*(x-xclamp)
        else:
            #inside
            res = np.zeros(len(x))
            res[imindist] = iminsign
            return res

            
class LambdaSet(Set):
    """Given some standalone function fcontains(x) which determines
    membership, produces a Set object.

    Optionally, a function fsample() can be provided to sample from the
    set."""
    def __init__(self,fcontains,fsample=None):
        self.fcontains = fcontains
        self.fsample = fsample
    def sample(self):
        if self.fsample:
            return self.fsample()
        return None
    def contains(self,x):
        return self.fcontains(x)


class MultiSet(Set):
    """A cartesian product of sets"""
    def __init__(self,*components):
        self.components = components
    def __str__(self):
        return ' x '.join(str(c) for c in self.components)
    def dimension(self):
        return sum(c.dimension() for c in self.components)
    def bounds(self):
        cbounds = [c.bounds() for c in self.components]
        if any(c==None for c in cbounds): return None
        bmin,bmax = zip(*cbounds)
        return self.join(bmin),self.join(bmax)
    def split(self,x):
        i = 0
        res = []
        for c in self.components:
            d = c.dimension()
            res.append(x[i:i+d])
            i += d
        return res
    def join(self,xs):
        return sum(xs,[])
    def contains(self,x):
        for (c,xi) in zip(self.components,self.split(x)):
            if not c.contains(xi):
                return False
        return True
    def sample(self):
        return self.join(c.sample() for c in self.components)
    def project(self,x):
        return self.join(c.project(xi) for c,xi in zip(self.components,self.split(x)))
    def signedDistance(self,x):
        return max(c.signedDistance(xi) for (c,xi) in zip(self.components,self.split(x)))
    def signedDistance_gradient(self,x):
        xis = self.split(x)
        pieces = [np.zeros(len(xi)) for xi in xis]
        imin = max((c.signedDistance(xi),i) for i,(c,xi) in enumerate(zip(self.components,xis)))[1]
        pieces[imin] = self.components[imin].signedDistance_gradient(xis[imin])
        return np.hstack(pieces)
