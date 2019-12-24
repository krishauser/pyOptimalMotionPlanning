"""Helper for K-nearest neighbor queries"""

from __future__ import print_function,division
from six import iteritems
from builtins import range

class KNearestResult:
    def __init__(self,k):
        assert k > 0
        self.items = [None]*k
        self.distances = [float('inf')]*k
        self.imin = 0
        self.imax = 0
    def tryadd(self,item,distance):
        if distance < self.distances[self.imax]:
            self.distances[self.imax] = distance
            self.items[self.imax] = item
            if distance < self.distances[self.imin]:
                self.imin = self.imax
            #update imin
            for i in range(len(self.items)):
                if self.distances[i] > self.distances[self.imax]:
                    self.imax = i
    def minimum_distance(self):
        return self.distances[self.imin]
    def maximum_distance(self):
        return self.distances[self.imax]
    def sorted_items(self):
        sorted_res = sorted([(d,i) for (i,d) in zip(self.items,self.distances) if d!=float('inf')])
        return [v[1] for v in sorted_res]
