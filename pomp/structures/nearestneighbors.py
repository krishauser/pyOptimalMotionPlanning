from __future__ import print_function,division
from six import iteritems
from builtins import range

from . import kdtree
from .knn import *

#set this to true if you wish to double-check the results of the kd tree
check_kdtree = False

infty = float('inf')

class NearestNeighbors:
    def __init__(self,metric,method='bruteforce'):
        self.metric = metric
        self.method = method
        if self.method == 'kdtree':
            self.kdtree = kdtree.KDTree(self.metric)
            if check_kdtree:
                self.checker = NearestNeighbors(self.metric)
                print("Debugging: Double checking KD-tree with nearest neighbors")
        else:
            self.nodes = []

    def reset(self):
        if self.method == 'kdtree':
            self.kdtree = kdtree.KDTree(self.metric)
            if check_kdtree: self.checker = NearestNeighbors(self.metric)
        else:
            self.nodes = []

    def add(self,point,data=None):
        """Adds a point with an associated datum."""
        if self.method == 'kdtree':
            self.kdtree.add(point,data)
            self.kdtree.rebalance()
            if check_kdtree: self.checker.add(point,data)
        else:
            self.nodes.append((point,data))

    def remove(self,point,data=None):
        """Removes a point, optionally matching the data too.
        Time is O(nearest).  Returns the number of points removed.
        (TODO: can only be 0 or 1 at the moment)."""
        if self.method == 'kdtree':
            res = self.kdtree.remove(point,data)
            if check_kdtree:
                cres = self.checker.remove(point,data)
                if cres != res:
                    raise ValueError("KD-tree did not remove the correct numer of points")
            if res == 0:
                print("KDTree: Unable to remove",point,"does not exist")
            return res
        else:
            for i,(p,pd) in enumerate(self.nodes):
                if p == point and (data == None or pd==data):
                    del self.nodes[i]
                    return 1
            print("ERROR REMOVING POINT FROM BRUTE-FORCE NN STRUCTURE")
            for p,pd in self.nodes:
                print(p,pd)
        return 0
            

    def set(self,points,datas=None):
        """Sets the point set to a list of points."""
        if datas==None:
            datas = [None]*len(points)
        if hasattr(self,'kdtree'):
            print("Resetting KD tree...")
            self.kdtree.set(points,datas)
            if check_kdtree: self.checker.set(points,datas)
        else:
            self.nodes = list(zip(points,datas))

    def nearest(self,pt,filter=None):
        """Nearest neighbor lookup, possibly with filter"""
        if self.method == 'kdtree':
            res = self.kdtree.nearest(pt,filter)
            if check_kdtree: 
                rescheck = self.checker.nearest(pt,filter)
                if res != rescheck:
                    print("KDTree nearest(",pt,") error",res,"should be",rescheck)
                    print(self.metric(res[0],pt))
                    print(self.metric(rescheck[0],pt))
            return res
        else:
            #brute force
            res = None    
            dbest = infty
            for p,data in self.nodes:
                d = self.metric(p,pt)
                if d < dbest and (filter == None or not filter(p,data)):
                    res = (p,data)
                    dbest = d
            return res

    def knearest(self,pt,k,filter=None):
        """K-nearest neighbor lookup, possibly with filter"""
        if self.method == 'kdtree':
            res = self.kdtree.knearest(pt,k,filter)
            if check_kdtree: 
                rescheck = self.checker.knearest(pt,k,filter)
                if res != rescheck:
                    print("KDTree knearest(",pt,") error",res,"should be",rescheck)
                    print(self.metric(res[0][0],pt))
                    print(self.metric(rescheck[0][0],pt))
            return res
        else:
            #brute force
            res = KNearestResult(k)
            for p,data in self.nodes:
                if (filter == None or not filter(p,data)):
                    d = self.metric(p,pt)
                    res.tryadd(d,(p,data))
            return res.sorted_items()
    
    def neighbors(self,pt,radius):
        """Range query, all points within pt.  Filtering can be done
        afterward by the user."""
        if self.method == 'kdtree':
            res = self.kdtree.neighbors(pt,radius)
            if check_kdtree: 
                rescheck = self.checker.neighbors(pt,radius)
                if len(res) != len(rescheck):
                    print("KDTree neighbors(",pt,",",radius,") error",res,"should be",rescheck)
                else:
                    for r in res:
                        if r not in rescheck:
                            print("KDTree neighbors(",pt,",",radius,") error",res,"should be",rescheck)
                            break

            return res
        else:
            #brute force
            res = []
            for p,data in self.nodes:
                d = self.metric(p,pt)
                if d < radius:
                    res.append((p,data))
            return res
      

