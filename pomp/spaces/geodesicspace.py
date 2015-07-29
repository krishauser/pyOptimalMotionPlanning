from klampt import vectorops
import math

class GeodesicSpace:
    def dimension(self):
        raise NotImplementedError()
    def distance(self,a,b):
        return vectorops.distance(a,b)
    def interpolate(self,a,b,u):
        return vectorops.interpolate(a,b,u)
    def difference(self,a,b):
        """For Lie groups, returns a difference vector that, when integrated
        would get to a from b.  In Cartesian spaces it is a-b."""
        return vectorops.sub(a,b)
    def integrate(self,x,d):
        """For Lie groups, returns the point that would be arrived at via
        integrating the difference vector d starting from x.  Must satisfy
        the relationship a = integrate(b,difference(a,b)). In Cartesian
        spaces it is x+d"""
        return vectorops.add(x,d)

class CartesianSpace(GeodesicSpace):
    """The standard geodesic on R^d"""
    def __init__(self,d):
        self.d = d
    def dimension(self):
        return self.d

class MultiGeodesicSpace:
    """This forms the cartesian product of one or more GeodesicSpace's.
    Distances are simply added together."""
    def __init__(self,*components):
        self.components = components
        self.componentWeights = [1]*len(self.components)
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
    def distance(self,a,b):
        i = 0
        res = 0.0
        for c,w in zip(self.components,self.componentWeights):
            d = c.dimension()
            res += (c.distance(a[i:i+d],b[i:i+d])**2)*w
            i += d
        return math.sqrt(res)
    def interpolate(self,a,b,u):
        i = 0
        res = [0]*len(a)
        for c in self.components:
            d = c.dimension()
            res[i:i+d] = c.interpolate(a[i:i+d],b[i:i+d],u)
            i += d
        return res
    def difference(self,a,b):
        i = 0
        res = [0]*len(a)
        for c in self.components:
            d = c.dimension()
            res[i:i+d] = c.difference(a[i:i+d],b[i:i+d])
            i += d
        return res
    def integrate(self,x,diff):
        i = 0
        res = [0]*len(x)
        for c in self.components:
            d = c.dimension()
            res[i:i+d] = c.difference(x[i:i+d],diff[i:i+d])
            i += d
        return res
