from __future__ import print_function,division
from builtins import range
from six import iteritems

import math
from ..klampt import vectorops,trajectory
from .geodesicspace import GeodesicSpace

class Interpolator:
    """A base class for an interpolating curve"""
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def length(self):
        """Evaluates the length of the interpolator"""
        raise NotImplementedError()
    def eval(self,u):
        """Evalutes the interpolating point for u in the range [0,1]"""
        raise NotImplementedError()
    def start(self):
        return self.a
    def end(self):
        return self.b
    def split(self,u):
        """Return a pair of interpolators split at self.eval(u)"""
        raise NotImplementedError()

class LinearInterpolator(Interpolator):
    def length(self):
        return vectorops.distance(self.a,self.b)
    def eval(self,u):
        return vectorops.interpolate(self.a,self.b,u)
    def split(self,u):
        mid = self.eval(u)
        return (LinearInterpolator(self.a,mid),LinearInterpolator(mid,self.b))

class GeodesicInterpolator(Interpolator):
    def __init__(self,a,b,space):
        self.a = a
        self.b = b
        self.space = space
    def length(self):
        return self.space.distance(self.a,self.b)
    def eval(self,u):
        return self.space.interpolate(self.a,self.b,u)
    def split(self,u):
        mid = self.eval(u)
        return (GeodesicInterpolator(self.a,mid,space),GeodesicInterpolator(mid,self.b,space))

class PathInterpolator(Interpolator):
    def __init__(self,edges):
        self.edges = edges
    def length(self):
        return sum([e.length() for e in self.edges])
    def eval(self,u):
        uk = u*len(self.edges)
        k = int(math.floor(uk))
        s = uk - k
        if k < 0: return self.start()
        if k >= len(self.edges): return self.end()
        return self.edges[k].eval(s)
    def start(self):
        return self.edges[0].start()
    def end(self):
        return self.edges[-1].end()
    def split(self,u):
        uk = u*len(self.edges)
        k = int(math.floor(uk))
        s = uk - k
        raise NotImplementedError("TODO")


class PiecewiseLinearInterpolator(Interpolator):
    def __init__(self,path,times=None,geodesic=None):
        Interpolator.__init__(self,path[0],path[-1])
        if geodesic != None:
            self.geodesic = geodesic
        else:
            self.geodesic = GeodesicSpace()
        if times != None:
            self.path = path
            self.trajectory = trajectory.Trajectory(times,path)
            if times[0] != 0 or times[-1] != 1:
                raise ValueError("PiecewiseLinearInterpolator must have time range [0,1]")
        else:
            self.path = path
            self.trajectory = None
    def length(self):
        l = 0
        for i in range(len(self.path)-1):
            l += self.geodesic.distance(self.path[i],self.path[i+1])
        return l
    def eval(self,u):
        if self.trajectory != None:
            return self.trajectory.eval(u)
        else:
            if u <= 0: return self.path[0]
            if u >= 1: return self.path[-1]
            k = u*(len(self.path)-1)
            s = k-math.floor(k)
            i = int(math.floor(k))
            return self.geodesic.interpolate(self.path[i],self.path[i+1],s)

class LambdaInterpolator(Interpolator):
    """A helper that takes a function feval(u) that
    interpolates between 0 and 1 and returns an interpolator object."""
    def __init__(self,feval,space=None,lengthDivisions=0):
        self.feval = feval
        self.space = space
        self.lengthDivisions = lengthDivisions
        Interpolator.__init__(self,feval(0),feval(1))
    def length(self):
        if self.lengthDivisions == 0:
            if self.space == None:
                return metric.euclidean(self.a,self.b)
            return self.space.distance(self.a,self.b)
        else:
            L = 0
            p = self.a
            for i in range(self.lengthDivisions):
                n = self.eval(float(i+1)/self.lengthDivisions)
                L += self.space.distance(p,n)
                p = n
            return L
    def eval(self,u):
        return self.feval(u)
    def split(self,u):
        return (LambdaInterpolator(lambda s:self.feval(u*s),self.space), \
                LambdaInterpolator(lambda s:self.feval(u + (1.0-u)*s),self.space))


class MultiInterpolator(Interpolator):
    """Cartesian product of multiple interpolators"""
    def __init__(self,*args,**kwargs):
        self.components = args
        if 'weights' in kwargs:
            self.componentWeights = kwargs['weights']
        else:
            self.componentWeights = None
    def length(self):
        if self.componentWeights != None:
            return sum(c.length() for c in self.components)
        else:
            return sum(c.length()*w for c,w in zip(self.components,self.componentWeights))
    def eval(self,u):
        return sum([c.eval(u)  for c in self.components], [])
    def start(self):
        return sum([c.start()  for c in self.components], [])
    def end(self):
        return sum([c.end()  for c in self.components], [])
    def split(self,u):
        splits = [c.split(u) for c in self.components]
        return (MultiInterpolator([s[0] for s in splits],weights=self.componentWeights),
                MultiInterpolator([s[1] for s in splits],weights=self.componentWeights))
