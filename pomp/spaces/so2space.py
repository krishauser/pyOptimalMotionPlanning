from ..klampt import so2
from .configurationspace import *
import random
import math

class SO2Geodesic(GeodesicSpace):
    def dimension(self):
        return 1
    def distance(self,a,b):
        return abs(so2.diff(a[0],b[0]))
    def interpolate(self,a,b,u):
        return [so2.interp(a[0],b[0],u)]
    def difference(self,a,b):
        return [so2.diff(a[0],b[0])]
    def integrate(self,x,d):
        return [so2.normalize(x[0]+d[0])]

class SO2Space(GeodesicConfigurationSpace):
    """Space of angles [0,2pi) supporting proper wrapping"""
    def __init__(self):
        self.geodesic = SO2Geodesic()
    def bounds(self):
        return [0],[math.pi*2]
    def sample(self):
        return [random.uniform(0,math.pi*2)]

