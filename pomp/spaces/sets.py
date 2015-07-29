import math
import random

class Set:
    def dimension(self):
        """Returns the number of entries of an element of this set"""
        try:
            x = self.sample()
            return len(x)
        except NotImplementedError:
            raise NotImplementedError()
    def bounds(self):
        """Returns a pair (bmin,bmax) giving an axis-aligned bounding box
        on items in this set.  Return None if no bound can be determined."""
        raise NotImplementedError()
    def sample(self):
        """Sample a random value from the set"""
        raise NotImplementedError()
    def contains(self,x):
        raise NotImplementedError()

class FiniteSet(Set):
    def __init__(self,items):
        self.items = items
    def bounds(self):
        bmin = self.items[0][:]
        bmax = self.items[0][:]
        for v in self.items[1:]:
            for i in xrange(len(v)):
                if v[i] < bmin[i]: bmin[i] = v[i]
                elif v[i] > bmax[i]: bmax[i] = v[i]
        return (bmin,bmax)
    def sample(self):
        return random.choice(self.items)
    def contains(self,x):
        return x in self.items

class BoxSet(Set):
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
