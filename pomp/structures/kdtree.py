from __future__ import print_function,division
from six import iteritems

import math
from .knn import *
from ..spaces import metric
from ..klampt import so2

#If using a non-Euclidean space, the kd tree should account for
#it.  Rather than setting the the KDTree partitionFn and minDistance
#functions by hand, you can set one of these to true and the space will
#be accounted for if it has SO2 or SE2 in its leading elements.
SO2_HACK = False
SE2_HACK = False

class Node:
    def __init__(self,points,splitdim=0):
        """Arguments:
        - points: a list of (point,data) pairs
        - splitdim: the default split dimension.
        """
        assert isinstance(points,(list,tuple))
        self.points = points
        self.splitdim = splitdim
        self.depth = 0
        self.splitvalue = None
        self.left = None
        self.right = None

class KDTree:
    """
    Attributes:
    - maxPointsPerNode: allows up to this number of points in a single K-D tree
      node.
    - partitionFn: a function (d,value,x) that returns -1,0, or 1
      depending on whether the point is on the positive, negative, or on the
      plane with split value value on the d'th dimension.  By default, it's
      sign(x[d] - value).
    - minDistanceFn: a function (d,value,x) that returns the minimum
      distance to the partition plane on the d'th dimension.  By default, it's
      |x[d] - value| for the euclidean metric, and otherwise it's
      metric(set(x,d,value),x[d]) where set(x,d,value) is a point equal
      to x except for it's d'th entry set to value.  This latter case works for
      all L-p norms, weighted or unweighted.
    """
    def __init__(self,distanceMetric=metric.euclideanMetric):
        self.root = None
        self.metric = distanceMetric
        self.partitionFn = lambda d,value,x: math.copysign(1,x[d]-value)
        self.minDistanceFn = lambda d,value,x: abs(x[d]-value)
        if self.metric != metric.euclideanMetric:
            def minDistFn(d,value,x):
                tempp = x[:]
                tempp[d] = value

                #HACK: for pendulum
                if SO2_HACK and d==0:
                    tempp2 = x[:]
                    tempp2[d] = math.pi+value
                    return min(self.metric(tempp,x),self.metric(tempp2,x))
                #HACK: for SE2
                if SE2_HACK and d==2:
                    tempp2 = x[:]
                    tempp2[d] = math.pi+value
                    return min(self.metric(tempp,x),self.metric(tempp2,x))
                return self.metric(tempp,x)
            def partitionFn(d,value,x):
                #HACK: for pendulum
                if SO2_HACK and d == 0:
                    math.copysign(1,so2.diff(x[d],value))
                #HACK: for SE2
                if SE2_HACK and d == 2:
                    math.copysign(1,so2.diff(x[d],value))
                return math.copysign(1,x[d]-value)
            self.minDistanceFn = minDistFn
            self.partitionFn = partitionFn

        self.maxPointsPerNode = 10
        self.maxDepth = 0
        self.numNodes = 0

    def locate(self,point):
        """Find the Node which contains a point"""
        return self._locate(self.root,point)

    def _locate(self,node,point):
        if node.splitvalue == None: return node
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0: return self._locate(node.left,point)
        else: return self._locate(node.right,point)

    def set(self,points,data):
        """Set the KD tree to contain a list of points.  O(log(n)*n*d)
        running time."""
        self.maxDepth = 0
        self.numNodes = 1
        self.root = Node(list(zip(points,data)))
        self.recursive_split(self.root,optimize=True)

    def recursive_split(self,node,force=False,optimize=False):
        """Recursively splits the node along the best axis.
        - node: the node to split
        - force: true if you want to force-split this node.  Otherwise,
          uses the test |node.points| > maxPointsPerNode to determine
          whether to split.
        - optimize: true if you want to select the split dimension with
          the widest range.
        Returns the depth of the subtree if node is split, 0 if not split.
        """
        if not force and len(node.points) <= self.maxPointsPerNode:
            return 0
        if len(node.points)==0:
            return 0
        if node.left != None:
            #already split
            raise RuntimeError("Attempting to split node already split")
        d = len(node.points[0][0])
        vmin,vmax = 0,0
        if not optimize:
            #just loop through the dimensions
            for i in range(d):
                vmin = min(p[0][node.splitdim] for p in node.points)
                vmax = max(p[0][node.splitdim] for p in node.points)
                if vmin != vmax:
                    break
                #need to choose a new split dimension
                node.splitdim = (node.splitdim+1)%d
        else:
            rangemax = (0,0)
            dimmax = 0
            for i in range(d):
                vmin = min(p[0][i] for p in node.points)
                vmax = max(p[0][i] for p in node.points)
                if vmax-vmin > rangemax[1]-rangemax[0]:
                    rangemax = (vmin,vmax)
                    dimmax = i
            node.splitdim = dimmax
            vmin,vmax = rangemax
        if vmin == vmax:
            #all points are equal, don't split (yet)
            return 0
        node.splitvalue = (vmin+vmax)*0.5
        leftpts = []
        rightpts = []
        for p in node.points:
            if self.partitionFn(node.splitdim,node.splitvalue,p[0]) < 0: leftpts.append(p)
            else: rightpts.append(p)
        if len(leftpts) == 0 or len(rightpts) == 0:
            #may have numerical error
            node.splitvalue = None
            return 0
        node.left = Node(leftpts,(node.splitdim+1)%d)
        node.right = Node(rightpts,(node.splitdim+1)%d)
        node.left.depth = node.depth+1
        node.right.depth = node.depth+1
        self.numNodes += 2
        self.maxDepth = max(self.maxDepth,node.depth+1)
        d1 = self.recursive_split(node.left,force=False,optimize=optimize)
        d2 = self.recursive_split(node.right,force=False,optimize=optimize)
        node.points = []
        return 1+max(d1,d2)

    def add(self,point,data):
        """Add a point to the KD tree (O(log(n)) running time)"""
        if self.root == None:
            self.root = Node([(point,data)])
            self.maxDepth = 0
            self.numNodes = 1
            return self.root
        else:
            node = self.locate(point)
            node.points.append((point,data))
            if self.recursive_split(node,optimize=False):
                return self._locate(node,point)
            else:
                return node

    def _locate_with_parent(self,node,point):
        if node.splitvalue == None: return (node,None)
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0:
            n,p = self._locate_with_parent(node.left,point)
            if p==None: return n,node
            return n,p
        else:
            n,p = self._locate_with_parent(node.right,point)
            if p==None: return n,node
            return n,p


    def remove(self,point,data=None):
        """Removes the point from the KD-tree.  If data is given, then the
        data member is checked for a match too.  Returns the number of points
        removed. (TODO: can only be 0 or 1 at the moment)"""
        n,parent = self._locate_with_parent(self.root,point)
        if n==None: return 0
        found = False
        for i,p in enumerate(n.points):
            if point == p[0] and (data==None or data==p[1]):
                del n.points[i]
                found = True
                break
        if len(n.points)==0:
            #merge siblings back up the tree?
            if parent != None:
                if parent.left.splitvalue == None and parent.right.splitvalue == None:
                    if parent.left == n:
                        parent.points = parent.right.points
                        parent.left = parent.right = None
                    else:
                        assert parent.right==n
                        parent.points = parent.left.points
                        parent.left = parent.right = None
                    parent.splitvalue = None
        if found:
            return 1
        return 0

    def rebalance(self,force=False):
        dorebalance = force
        if not force:
            idealdepth = math.log(self.numNodes)
            #print("ideal depth",idealdepth,"true depth",self.maxDepth)
            if self.maxDepth > idealdepth*10:
                dorebalance = True
        if not dorebalance:
            return False
        print("Rebalancing KD-tree...")
        points = []
        def recurse_add_points(node):
            points += node.points
            if node.left: recurse_add_points(node.left)
            if node.right: recurse_add_points(node.right)
        recurse_add_points(self.root)
        self.set(zip(*points))
        print("Done.")
        return True

    def _nearest(self,node,x,dmin,filter=None):
        if node.splitvalue == None:
            #base case, it's a leaf
            closest = None
            for p in node.points:
                if filter != None and filter(*p):
                    continue
                d = self.metric(p[0],x)
                if d < dmin:
                    closest = p
                    dmin = d
            return (closest,dmin)
        #recursive case, it's a parent
        dhi = 0
        dlo = 0
        if self.partitionFn(node.splitdim,node.splitvalue,x) < 0:
            dlo = 0
            dhi = self.minDistanceFn(node.splitdim,node.splitvalue,x)
        else:
            dhi = 0
            dlo = self.minDistanceFn(node.splitdim,node.splitvalue,x)

        if dhi > dmin:  #only check left
            (lclosest,ld) = self._nearest(node.left,x,dmin,filter)
            return (lclosest,ld)
        elif dlo > dmin: #only check right
            (rclosest,rd) = self._nearest(node.right,x,dmin,filter)
            return (rclosest,rd)
        else:
            first,second = node.left,node.right
            if dlo > dhi:
                first,second = second,first
                dlo,dhi = dhi,dlo
            #check the closest first
            closest = None
            (fclosest,fd) = self._nearest(first,x,dmin,filter)
            if fd < dmin:
                #assert fclosest != None
                #assert fd == self.metric(fclosest[0],x)
                closest,dmin=fclosest,fd
            if dhi < dmin: #check if should prune second or not
                #no pruning, check the second next
                (sclosest,sd) = self._nearest(second,x,dmin,filter)
                if sd < dmin:
                    #assert sclosest != None
                    #assert sd == self.metric(sclosest[0],x)
                    closest,dmin=sclosest,sd
            return (closest,dmin)

    def nearest(self,x,filter=None):
        """Nearest neighbor query:
        Returns the (point,data) pair in the tree closest to the point x"""
        if self.root == None: return []
        closest,dmin = self._nearest(self.root,x,float('inf'),filter)
        if closest is None:
            input()
        return closest

    def _knearest(self,node,x,res,filter=None):
        if node.splitvalue == None:
            #base case, it's a leaf
            for p in node.points:
                if filter != None and filter(*p):
                    continue
                d = self.metric(p[0],x)
                res.tryadd(p,d)
            return res
        #recursive case, it's a parent
        dhi = 0
        dlo = 0
        if self.partitionFn(node.splitdim,node.splitvalue,x) < 0:
            dlo = 0
            dhi = self.minDistanceFn(node.splitdim,node.splitvalue,x)
        else:
            dhi = 0
            dlo = self.minDistanceFn(node.splitdim,node.splitvalue,x)

        if dhi > res.maximum_distance():  #only check left
            res = self._knearest(node.left,x,res,filter)
            return res
        elif dlo > res.maximum_distance(): #only check right
            res = self._knearest(node.right,x,res,filter)
            return res
        else:
            first,second = node.left,node.right
            if dlo > dhi:
                first,second = second,first
                dlo,dhi = dhi,dlo
            #check the closest first
            closest = None
            res = self._knearest(first,x,res,filter)
            if dhi < res.maximum_distance(): #check if should prune second
                #no pruning, check the second next
                res = self._knearest(second,x,res,filter)
            return res


    def knearest(self,x,k,filter=None):
        """K-nearest neighbor query:
        Returns the [(point1,data1),...,(pointk,datak)] in the tree
        that are closest to the point x. Results are sorted by distance."""
        if self.root == None: return []
        res = self._knearest(self.root,x,KNearestResult(k),filter)
        return res.sorted_items()

    def _neighbors(self,node,x,rad,results):
        if node.splitvalue == None:
            #base case, it's a leaf
            for p in node.points:
                d = self.metric(p[0],x)
                if d <= rad:
                    results.append(p)
            return
        #recursive case, it's a parent
        dhi = 0
        dlo = 0
        if self.partitionFn(node.splitdim,node.splitvalue,x) < 0:
            dlo = 0
            dhi = self.minDistanceFn(node.splitdim,node.splitvalue,x)
        else:
            dhi = 0
            dlo = self.minDistanceFn(node.splitdim,node.splitvalue,x)

        if dhi <= rad:  #check right
            self._neighbors(node.right,x,rad,results)
        if dlo <= rad: # check left
            self._neighbors(node.left,x,rad,results)

    def neighbors(self,x,rad):
        """Distance neighbor query:
        Returns the list of (point,data) pairs in the tree within distance
        rad to the point x"""
        if self.root == None: return []
        retval = []
        self._neighbors(self.root,x,rad,retval)
        return retval
