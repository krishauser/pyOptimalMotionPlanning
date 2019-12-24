from __future__ import print_function,division
from builtins import range
from six import iteritems

from ..spaces.sampler import *
from ..spaces.costspace import *
from ..spaces.edgechecker import EdgeChecker
import time
from ..spaces import metric
from ..structures.randomdict import RandomDict
from ..structures.nearestneighbors import *
from .profiler import *
from .helpers import *
import itertools

#rrtNumControlSampleIters = 10
rrtNumControlSampleIters = 1

#Set this to something non multiple of 10 because many test obstacles are
#aligned to a 10x10 grid
estDefaultResolution = 13
#Hierarchical density estimation 
estHierarchyLevels = 2
#when caching is on, this can be set quite high because only a few extension
#samples are being drawn per iteration.  Drawback: re-weighting frequency
#becomes an issue
#OK value for no cache
#estNumExtensionSamples = 10
#small cache
estNumExtensionSamples = 50
#large cache
#estNumExtensionSamples = 1000
#Tests seem to indicate that cached extensions aren't very useful
#at least on simple problems.
#Suspicion: because we're pre-checking edge feasibility, narrow passages are
#being sampled less frequently
estUseCachedExtensions = True
#estUseCachedExtensions = False
estPrecheckExtensions = True
#estPrecheckExtensions = False
estCacheReweightFrequency = 10
estNumCachedExtensionDrops = 2
#TEST: add a "bonus extension" to cache after a successful extension
#this might help wiggle into narrow passages
estLetItRoll = True
estNumLetItRollSamples = 50



class ControlSelector:
    """A function that produces a control u that tries to steer a
    state x to another state xdesired."""
    def select(self,x,xdesired):
        raise NotImplementedError()


class RandomControlSelector(ControlSelector):
    """A ControlSelector that randomly samples numSamples controls
    and finds the one that is closest to the destination, according
    to a given metric."""
    def __init__(self,controlSpace,metric,numSamples):
        self.controlSpace = controlSpace
        self.metric = metric
        self.numSamples = numSamples

    def select(self,x,xdesired):
        ubest = None
        #do we want to eliminate extensions that do not improve the metric
        #from the starting configuration?  experiments suggest no 5/6/2015
        #dbest = self.metric(x,xdesired)
        dbest = infty
        U = self.controlSpace.controlSet(x)
        if self.numSamples == 1:
            u = U.sample()
            return u if U.contains(u) else None
        for iters in range(self.numSamples):
            u = U.sample()
            if U.contains(u):
                xnext = self.controlSpace.nextState(x,u)
                d = self.metric(xnext,xdesired)
                if d < dbest:
                    dbest = d
                    ubest = u
        return ubest


class KinematicControlSelector(ControlSelector):
    """A steering function for a kinematic control space"""
    def __init__(self,controlSpace,maxDistance=float('inf')):
        assert isinstance(controlSpace,ControlSpaceAdaptor)
        self.controlSpace = controlSpace
        self.cspace = controlSpace.configurationSpace()
        self.maxDistance = maxDistance

    def select(self,x,xdesired):
        d = self.cspace.distance(x,xdesired)
        if d > self.maxDistance:
            e = self.cspace.interpolator(x,xdesired)
            return e.eval(self.maxDistance/d)
        return xdesired


class KinematicCostControlSelector:
    """A steering function for a CostControlSpace on top of a kinematic
    control space"""
    def __init__(self,controlSpace,maxDistance=float('inf')):
        assert isinstance(controlSpace,CostControlSpace)
        assert isinstance(controlSpace.baseSpace,ControlSpaceAdaptor)
        self.controlSpace = controlSpace
        self.cspace = controlSpace.baseSpace.configurationSpace()
        self.maxDistance = maxDistance

    def select(self,x,xdesired):
        xb = x[:-1]
        xdb = xdesired[:-1]
        d = self.cspace.distance(xb,xdb)
        if d > self.maxDistance:
            e = self.cspace.interpolator(xb,xdb)
            return e.eval(self.maxDistance/d)
        return xdb


class Node:
    """A node of a kinodynamic tree"""
    def __init__(self,x,uparent=None,eparent=None):
        self.x = x
        self.uparent = uparent
        self.eparent = eparent
        self.parent = None
        self.children = []

    def destroy(self):
        """Call this to free up the memory stored by this sub tree."""
        if self.parent:
            self.parent.children.remove(self)
            self.parent = None
        for c in self.children:
            c.parent = None
            c.destroy()
        self.children = []

    def unparent(self):
        """Detatches this node from its parent"""
        if self.parent:
            self.parent.children.remove(self)
            self.parent = None

    def addChild(self,c):
        """Adds a child"""
        assert c.parent is None
        c.parent = self
        self.children.append(c)

    def setParent(self,p,uparent=None,eparent=None):
        """Sets the parent of this node to p"""
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = p
        self.uparent = uparent
        self.eparent = eparent
        p.children.append(self)

    def traverse(self,visitor):
        """Given a callback visitor(n), traverses the tree in DFS fashion.
        The visitor function can return False to prune traversal below a
        given node."""
        if visitor(self)==False:
            return
        for c in self.children:
            c.traverse(visitor)
        return


class TreePlanner:
    """A base class for kinodynamic tree planners"""
    def __init__(self):
        self.root = None
        self.nodes = []

    def destroy(self):
        """To be nice to the GC, call this to free up memory after you're
        done planning"""
        if self.root:
            self.root.destroy()
            self.root = None

    def setRoot(self,x):
        """Sets the root of the tree"""
        if self.root is not None:
           self.root.destroy()
        self.root = Node(x)
        self.nodes = [self.root]

    def addEdge(self,n,u,edge):
        """Adds an edge to the tree"""
        nnew = Node(edge.end(),u,edge)
        n.addChild(nnew)
        self.nodes.append(nnew)
        return nnew
        
    def getPath(self,n):
        """Returns a state-control pair ([x0,...,xn],[u1,...,un])"""
        pathToStart = []
        while n is not None:
            pathToStart.append(n)
            n = n.parent
        pathToGoal = pathToStart
        pathToGoal.reverse()
        return ([n.x for n in pathToGoal],[n.uparent for n in pathToGoal[1:]])
        
    def getRoadmap(self):
        """Returns a graph (V,E) where V contains states and E contains
        triples (i,j,u) where control u connnects V[i] to V[j]"""
        V = []
        E = []
        n = self.root
        if n is None:
            return (V,E)
        V.append(n.x)
        q = [(n,0)]
        while len(q) > 0:
            n,i = q.pop()
            for c in n.children:
                j = len(V)
                E.append((i,j,c.uparent))
                V.append(c.x)
                q.append((c,j))
        #what about ghost nodes not attached to tree?
        """
        nullnodes = 0
        for n in self.nodes:
            if n.x is not None:
                V.append(n.x)
            else:
                nullnodes += 1
        print("Null nodes:",nullnodes)
        """
        return (V,E)

    def getBestPath(self,obj,goal=None):
        """Given an objective function, finds the best path in the tree,
        optionally terminating in a goal Set.
        
        Returns a (xs,us,cost) triple"""
        def evalCost(n,costToCome):
            if len(n.children) == 0:
                if goal is None or goal.contains(n.x):
                    return costToCome + obj.terminal(n.x),n
                return float('inf'),n
            else:
                costbest = []
                for c in n.children:
                    c_edge = obj.incremental(n.x,c.uparent)
                    costbest.append(evalCost(c,costToCome + c_edge))
                return min(costbest)
        c,n = evalCost(self.root,0)
        xs,us = self.getPath(n)
        return xs,us,c


class RRT(TreePlanner):
    """The Rapidly-exploring Random Tree kinodynamic planner.

    Stores a tree of Nodes.  Expands the tree at random using the RRT strategy
    with a goal bias of probability pChooseGoal.

    Default controlSelector uses a RandomControlSelector with 10 samples.

    If you turn dynamicDomain = True, then the Adaptive Dynamic-Domain RRT
    planner of Jaillet et al 2005 is used.
    """
    def __init__(self,controlSpace,metric,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        TreePlanner.__init__(self)
        if not isinstance(controlSpace,ControlSpace):
            print("Warning, controlSpace is not a ControlSpace")
        if not isinstance(edgeChecker,EdgeChecker):
            print("Warning, edgeChecker is not an EdgeChecker")
        self.cspace = controlSpace.configurationSpace()
        if not isinstance(self.cspace,ConfigurationSpace):
            print("Warning, controlSpace.configurationSpace() is not a ConfigurationSpace")
        self.controlSpace = controlSpace
        self.metric = metric
        self.edgeChecker = edgeChecker
        self.goal = None
        self.goalSampler = None
        self.pChooseGoal = popdefault(params,'pChooseGoal',0.1)
        self.goalNodes = []
        self.configurationSampler = Sampler(self.cspace)
        numControlSamples = popdefault(params,'numControlSamples',rrtNumControlSampleIters)   
        self.controlSelector = RandomControlSelector(controlSpace,self.metric,numControlSamples)
        self.successBiasing = False
        self.dynamicDomain = popdefault(params,'dynamicDomain',False)
        if self.dynamicDomain:
            self.dynamicDomainInitialRadius = popdefault(params,'dynamicDomainInitialRadius',0.1)
            self.dynamicDomainInitialRadius = popdefault(params,'dynamicDomainGrowithParameter',0.5)
        nnmethod = popdefault(params,'nearestNeighborMethod','kdtree')
        self.nearestNeighbors = NearestNeighbors(self.metric,nnmethod)

        self.pruner = None
        self.stats = Profiler()
        self.numIters = self.stats.count('numIters')
        self.nextSampleList = []
        if len(params) != 0:
            print("Warning, unused params",params)
        
    def destroy(self):
        TreePlanner.destroy(self)
        self.goalNodes = []
        
    def reset(self):
        """Re-initializes the RRT to the same start / goal, clears the planning
        tree."""
        x0 = self.root.x
        goal = self.goal
        self.destroy()
        self.setBoundaryConditions(x0,goal)
        self.numIters.set(0)
	
    def setBoundaryConditions(self,x0,goal):
        """Initializes the tree from a start state x0 and a goal
        ConfigurationSubset.
        
        goal can be set to None to just explore.
        """
        self.setRoot(x0)
        self.root.numExpansionsAttempted = 0
        self.root.numExpansionsSuccessful = 0
        self.goal = goal
        if goal is not None:
            if isinstance(goal,(list,tuple)):
                self.goal = SingletonSubset(self.cspace,goal)
            self.goalSampler = SubsetSampler(self.cspace,self.goal)
        self.nearestNeighbors.reset()
        self.nearestNeighbors.add(x0,self.root)
        
    def setConfigurationSampler(self,sampler):
        self.configurationSampler = sampler
        
    def setControlSelector(self,selector):
        self.controlSelector = selector
        
    def planMore(self,iters):
        for n in range(iters):
            self.numIters += 1
            n = self.expand()
            if n is not None and self.goal is not None:
                if self.goal.contains(n.x):
                    self.goalNodes.append(n)
                    return True
        return False
        
    def expand(self):
        """Expands the tree via the RRT technique.  Returns the new node
        or None otherwise."""
        if len(self.nextSampleList)==0:
            if self.goalSampler and random.uniform(0.0,1.0) < self.pChooseGoal:
                xrand = self.goalSampler.sample()
            else:
                xrand = self.configurationSampler.sample()
            if not self.cspace.feasible(xrand):
                return None
        else:
            xrand = self.nextSampleList.pop(0)
        #self.stats.stopwatch('pickNode').begin()
        nnear = self.pickNode(xrand)
        #self.stats.stopwatch('pickNode').end()
        if nnear is None:
            #self.stats.count('pickNodeFailure').add(1)
            return None
        #self.stats.stopwatch('selectControl').begin()
        nnear.numExpansionsAttempted += 1
        u = self.controlSelector.select(nnear.x,xrand)
        #self.stats.stopwatch('selectControl').end()
        #print("Expanding",nnear.x,"toward",xrand,"selected control",u)
        if u is None:
            #do we want to adjust the dynamic domain?
            if self.dynamicDomain:
                if hasattr(nnear,'ddRadius'):
                    nnear.ddRadius *= (1.0-self.dynamicDomainGrowthParameter)
                else:
                    nnear.ddRadius = self.dynamicDomainInitialRadius
            self.stats.count('controlSelectionFailure').add(1)
            return None
        #self.stats.stopwatch('edgeCheck').begin()
        edge = self.controlSpace.interpolator(nnear.x,u)
        if not self.edgeChecker.feasible(edge):
            #self.stats.stopwatch('edgeCheck').end()
            if self.dynamicDomain:
                if hasattr(nnear,'ddRadius'):
                    nnear.ddRadius *= (1.0-self.dynamicDomainGrowthParameter)
                else:
                    nnear.ddRadius = self.dynamicDomainInitialRadius
            #self.stats.count('infeasibleEdges').add(1)
            return None
        #self.stats.stopwatch('edgeCheck').end()
        #feasible edge, add it
        if self.dynamicDomain:
            if hasattr(nnear,'ddRadius'):
                nnear.ddRadius *= (1.0+self.dynamicDomainGrowthParameter)
        nnew = self.addEdge(nnear,u,edge)
        if self.prune(nnew):
            nnew.destroy()
            self.nodes.pop()
            return None
        self.nearestNeighbors.add(nnew.x,nnew)
        nnear.numExpansionsSuccessful += 1
        nnew.numExpansionsAttempted = 0
        nnew.numExpansionsSuccessful = 0
        return nnew
        
    def prune(self,node):
        """Overload this to add tree pruning.  Return True to prune a node"""
        if self.pruner:
            #if self.pruner(node):
            #    print("asking to prune node",node.x)
            return self.pruner(node)
        return False
        
    def pruneTree(self):
        """Prunes all branches of the tree that should be pruned according to
        the prune() function, updates all internal data structures."""
        #self.stats.stopwatch('pruneTree').begin()
        
        def pruneIt(n):
            newchildren = []
            delchildren = []
            for c in n.children:
                if self.prune(c) or not self.cspace.feasible(c.x):
                    delchildren.append(c)
                else:
                    newchildren.append(c)
            for c in delchildren:
                c.parent = None
                c.destroy()
            n.children = newchildren
            return True
            
        newNodes = []
        def addNodes(n):
            newNodes.append(n)
            assert not self.prune(self.root),"Root node is asked to be pruned... can't handle this case"
            
        self.root.traverse(pruneIt)
        self.root.traverse(addNodes)
        self.nodes = newNodes
        self.nearestNeighbors.set([n.x for n in self.nodes],self.nodes)
        #self.stats.stopwatch('pruneTree').end()
        
    def pickNode(self,xrand):
        """Picks a node closest to xrand.  If dynamicDomain is True,
        uses the radius associated with the node"""
        #setup nearest neighbor filters
        filters = [lambda pt,n: self.prune(n)]
        if self.dynamicDomain:
            filters.append(lambda pt,n:hasattr(n,'ddRadius') and self.metric(n.x,xrand) >= n.ddRadius)
        if self.successBiasing:
            filters.append(lambda pt,n: (random.random() > float(n.numExpansionsSuccessful+1) / float(n.numExpansionsAttempted+1)))
        #do the lookup
        res = self.nearestNeighbors.nearest(xrand,lambda pt,n:any(f(pt,n) for f in filters))
        if res is None:
            print("Uh... pickNode(",xrand,") returned None?")
            #print("Nearest:",self.nearestNeighbors.nearest(xrand)[0])
            return None
        n = res[1]
        if n.x is None:
            raise ValueError("Picked a node for expansion without a state, may have been deleted")
        #print("closest to",xrand,"=",n.x)
        return n
    
    def getPath(self,n=None):
        if n is None:
            if len(self.goalNodes)==0:
                return None
            return TreePlanner.getPath(self,self.goalNodes[0])
        return TreePlanner.getPath(self,n)



class EST(TreePlanner):
    """The Expansive Space Tree kinodynamic planner.

    Stores a tree of Nodes.  Expands the tree at random using the inverse
    density weighting strategy.
    """
    def __init__(self,controlSpace,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        TreePlanner.__init__(self)
        self.cspace = controlSpace.configurationSpace()
        self.controlSpace = controlSpace
        self.edgeChecker = edgeChecker
        self.goal = None
        self.goalNodes = []
        self.stats = Profiler()
        self.numIters = self.stats.count('numIters')
        self.pruner = None
        self.radius = popdefault(params,'densityEstimationRadius',0.1)
        self.nearestNeighbors = NearestNeighbors(self.cspace.distance,popdefault(params,'nearestNeighborMethod','kdtree'))
        if len(params) != 0:
            print("Warning, unused params",params)

    def destroy(self):
        TreePlanner.destroy(self)
        self.goalNodes = []

    def setBoundaryConditions(self,x0,goal):
        """Initializes the tree from a start state x0 and a goal
        ConfigurationSubset.
        
        goal can be set to None to just explore.
        """
        if self.root is not None:
            self.destroy()
        self.setRoot(x0)
        self.onAddNode(self.root)
        self.goal = goal
        if goal is not None:
            if isinstance(goal,(list,tuple)):
                self.goal = SingletonSubset(self.cspace,goal)

    def reset(self):
        self.setBoundaryConditions(self.root.x,self.goal)
        self.numIters.set(0)
        self.goalNodes = []
        if hasattr(self,'extensionCache'):
            del self.extensionCache

    def planMore(self,iters):
        for n in range(iters):
            self.numIters += 1
            n = self.expand()
            if n is not None and self.goal is not None:
                if self.goal.contains(n.x):
                    #print("new goal node",n.x)
                    self.goalNodes.append(n)
                    return True
        return False

    def expand(self):
        """Expands the tree via the EST technique.  Returns the new node
        or None otherwise."""
        #control sampling method
        global estNumExtensionSamples
        numNodeSamples = min(10+len(self.nodes),estNumExtensionSamples)
        numControlSamplesPerNode = 10
        #numNodeSamples = 1
        #Temp: test some probability of rejection?
        #extensions = [None]
        #weights = [1.0]
        cachedExtensions = estUseCachedExtensions
        if not cachedExtensions:
            weights = []
            extensions = []
        else:
            if not hasattr(self,'extensionCache'):
                weights = []
                extensions = []
                self.extensionCache = (weights,extensions)
            else:
                (weights,extensions) = self.extensionCache
            #clear cache occasionally
            if self.numIters.count % 100000 == 0:
                weights = []
                extensions = []
                self.extensionCache = (weights,extensions)
            #re-estimate density of existing extensions
            if self.numIters.count % estCacheReweightFrequency == 0:
                for i in range(len(weights)):
                    n = extensions[i][0]
                    edge = extensions[i][2]
                    de = self.density(edge.end())
                    weights[i] = 1.0/(1.0+de**2)
        for n in range(numNodeSamples - len(weights)):
            #self.stats.count('numExtensionSamples').add(1)
            nnear = self.pickNode()
            if nnear is None:
                return None
            #print("density",d0,"numsamples",numControlSamples)
            U = self.controlSpace.controlSet(nnear.x)
            #TEST: try sampling more extensions from sparse areas
            #d0 = self.density(nnear.x)
            #numControlSamples = int(1+10.0/(1.0+d0))
            #for i in range(numControlSamples):
            for i in range(numControlSamplesPerNode):
                u = U.sample()
                if U.contains(u):
                    edge = self.controlSpace.interpolator(nnear.x,u)
                    if self.pruneExtension(nnear,u):
                        continue
                    if not self.cspace.feasible(edge.end()):
                        continue
                    if estPrecheckExtensions:
                        #self.stats.count('numEdgeChecks').add(1)
                        if not self.edgeChecker.feasible(edge):
                            continue
                        de = self.density(edge.end())
                        extensions.append((nnear,u,edge))
                        # pick with probability inversely proportional to density
                        #weights.append(1.0/(1.0+de))
                        # pick with probability inversely proportional to density squared?
                        weights.append(1.0/(1.0+de**2))
                        #print("density",de)
                        #TEST? break on successful extension?
                        break
                    if len(weights) >= numNodeSamples:
                        break
            if len(weights) >= numNodeSamples:
                break
        assert len(extensions) == len(weights),"Uhhh internal implementation error?"
        if len(extensions)==0:
            #failed extension
            #TODO: penalize this node
            return None
        #print("weights",weights)
        i = sample_weighted(weights)
        #i = arg_max(weights)
        #print("picked node with density",self.density(extensions[i][2].end()))
        if extensions[i]==None:
            #this gives some probability to rejecting extensions
            #in highly dense regions
            return None
        n,u,edge = extensions[i]
        
        if cachedExtensions:
            #delete from cache
            weights[i] = weights[-1]
            extensions[i] = extensions[-1]
            weights.pop(-1)
            extensions.pop(-1)
            for k in range(estNumCachedExtensionDrops):
                if len(weights)==0: break
                #i = random.randint(0,len(weights)-1)
                i = arg_min(weights)
                weights[i] = weights[-1]
                extensions[i] = extensions[-1]
                weights.pop(-1)
                extensions.pop(-1)
        if not estPrecheckExtensions:
            #cached extensions are pre-checked for feasibility
            #self.stats.count('numEdgeChecks').add(1)
            if not self.edgeChecker.feasible(edge):
	        #TODO: penalize this node
                return None

        #feasible edge, add it
        nnew = self.addEdge(n,u,edge)
        if self.prune(nnew):
            nnew.destroy()
            self.nodes.pop()
            return None
        self.onAddNode(nnew)
        #"let-it-roll" try adding an extension from the newly added node
        if cachedExtensions and estLetItRoll:
            U = self.controlSpace.controlSet(nnew.x)
            for i in range(estNumLetItRollSamples):
                u = U.sample()
                if U.contains(u):
                    edge = self.controlSpace.interpolator(nnew.x,u)
                    if not self.cspace.feasible(edge.end()):
                        continue
                    if estPrecheckExtensions:
                        #self.stats.count('numEdgeChecks').add(1)
                        if not self.edgeChecker.feasible(edge):
                            continue
                    de = self.density(edge.end())
                    extensions.append((nnew,u,edge))
                    # pick with probability inversely proportional to density
                    #weights.append(1.0/(1.0+de))
                    # pick with probability inversely proportional to density squared?
                    weights.append(1.0/(1.0+de**2))
                    break
        return nnew

    def onAddNode(self,n):
        """Adds nodes' density contribution to nearby nodes"""
        #sigma of 4
        neighbors = self.nearestNeighbors.neighbors(n.x,self.radius*4.0)
        d = 1.0
        for xother,nother in neighbors:
            if nother is n: continue
            dist = self.cspace.distance(n.x,xother)
            d += math.exp(-(dist/self.radius)**2)
            nother.density += d
        n.density = d
        self.nearestNeighbors.add(n.x,n)

    def density(self,x):
        neighbors = self.nearestNeighbors.neighbors(x,self.radius*4.0)
        d = 0.0
        for nx,n in neighbors:
            d += math.exp(-(self.cspace.distance(nx,x)/self.radius)**2)
        return d

    def pickNode(self):
        """Picks a node according to inverse density weighting strategy"""
        return sample_weighted([1.0/n.density for n in self.nodes],self.nodes)
        
    def prune(self,node):
        """Overload this to add tree pruning.  Return True to prune a node"""
        if self.pruner:
            return self.pruner(node)
        return False

    def pruneExtension(self,n,u):
        """Overload this to add extension pruning.  n is the parent, u is the
        control leading from n.x"""
        return False

    def pruneTree(self):
        """Prunes all branches of the tree that should be pruned according to
        the prune() function, updates all internal data structures."""
        def pruneIt(n):
                newchildren = []
                delchildren = []
                for c in n.children:
                    if self.prune(c) or not self.cspace.feasible(c.x):
                        delchildren.append(c)
                    else:
                        newchildren.append(c)
                for c in delchildren:
                    c.parent = None
                    c.destroy()
                n.children = newchildren
                return True
        newNodes = []
        def addNodes(n):
            self.onAddNode(n)
            newNodes.append(n)
        assert not self.prune(self.root),"Root node is asked to be pruned... can't handle this case"
        self.root.traverse(pruneIt)
        self.root.traverse(addNodes) 
        self.nodes = newNodes
        
    def getPath(self,n=None):
        """Returns a path to the node n if specified, or to the first goal
        node by default."""
        if n is None:
            if len(self.goalNodes)==0:
                return None
            return TreePlanner.getPath(self,self.goalNodes[0])
        return TreePlanner.getPath(self,n)


class ESTWithProjections(EST):
    """The EST but with a faster density estimator and data structure
    update.
    """
    def __init__(self,controlSpace,edgeChecker,
                 **params):
        self.projectionBases = []
        self.projectionHashes = []
        self.projectionResolution = popdefault(params,'projectionResolution',estDefaultResolution)
        self.projectionHierarchyLevels = popdefault(params,'projectionHierarchyLevels',estHierarchyLevels)
        EST.__init__(self,controlSpace,edgeChecker,
                     **params)

    def destroy(self):
        EST.destroy(self)
        self.projectionBases = []
        self.projectionHashes = []

    def reset(self):
        self.projectionBases = []
        self.projectionHashes = []
        EST.reset(self)

    def generateDefaultBases(self,indices):
        #generate exhaustive bases for the indicated space dimensions
        #first, determine a scale factor
        try:
            bmin,bmax = self.controlSpace.configurationSpace().bounds()
            scale = [1.0/(b-a) for (a,b) in zip(bmin,bmax)]
            if isinstance(self.controlSpace,CostControlSpace):
                scale[-1] *= 2.0
            """
            if isinstance(self.controlSpace,CostControlSpace):
                if scale[-1] != 0:
                    scale[-1] = 1
            """
            print("EST projection hash scale",scale,"resolution",self.projectionResolution)
        except Exception:
            scale = [1]*self.controlSpace.configurationSpace().dimension()
            print("EST projection hash scale",scale,"resolution",self.projectionResolution)
        #now enumerate all size-4 subsets of the indices
        d = min(len(indices),4)
        self.projectionBases = []
        self.projectionHashes = []
        for element in itertools.combinations(indices,d):
            self.projectionBases.append([])
            for i in element:
                basis = {i:self.projectionResolution*scale[i]}
                self.projectionBases[-1].append(basis)
            self.projectionHashes.append(RandomDict())
        #add hierarchy
        baseCnt = len(self.projectionHashes)
        for level in range(1,self.projectionHierarchyLevels):
            for b in range(baseCnt):
                basis = []
                for base in self.projectionBases[b]:
                    scalebasis = dict((i,v*(1+level)) for i,v in iteritems(base))
                    basis.append(scalebasis)
                self.projectionBases.append(basis)
                self.projectionHashes.append(RandomDict())
        print("EST using",len(self.projectionBases),"projection bases")
        if self.root is not None:
            #need to re-add the elements of the tree
            def recursive_add(node):
                self.onAddNode(node)
                for c in node.children:
                    recursive_add(c)
            recursive_add(self.root)
        if hasattr(self,'extensionCache'):
            del self.extensionCache

    def onAddNode(self,n):
        """Adds nodes to hash functions"""
        if len(self.projectionBases)==0:
            self.generateDefaultBases(list(range(len(n.x))))
        for basis,bhash in zip(self.projectionBases,self.projectionHashes):
            #sparse dot products
            dp = [0.0]*len(basis)
            for i,basisvector in enumerate(basis):
                for (k,v) in iteritems(basisvector):
                    #experimental
                    #if isinstance(self.controlSpace,CostControlSpace) and k==len(n.x)-1:
                    #    #cost log transform
                    #    dp[i] += math.log(n.x[k]*v+1.0)*3.0
                    #else:
                    dp[i] += n.x[k]*v
            index = tuple([int(v) for v in dp])
            bhash.setdefault(index,[]).append(n)
        return

    def density(self,x):
        #c = 1.0
        #c = float('inf')
        c = 0.0
        baselevels = len(self.projectionBases)/self.projectionHierarchyLevels
        for index,(basis,bhash) in enumerate(zip(self.projectionBases,self.projectionHashes)):
            blevel = (int(index / baselevels) + 1)
            #sparse dot products
            dp = [0.0]*len(basis)
            for i,basisvector in enumerate(basis):
                for (k,v) in iteritems(basisvector):
                    #experimental: make high-cost cells larger
                    #if isinstance(self.controlSpace,CostControlSpace) and k==len(x)-1:
                    #    #cost log transform
                    #    dp[i] += math.log(x[k]*v+1.0)*3.0
                    #else:
                    dp[i] += x[k]*v
            index = tuple([int(v) for v in dp])
            cnt = len(bhash.get(index,[]))
            #here we've added to get an average density
            c += float(cnt)*pow(blevel,len(basisvector))
            #here we've multiplied to get a better sense of density
            #c *= float(cnt)/(len(self.nodes))
            #c = min(c,cnt)
        return c / len(self.projectionBases)
        #account for multiple counting of axes
        #power = (float(len(x))-1)/(3.0*len(self.projectionBases))
        #if len(self.projectionBases) > 1:
        #    print("Correction power",power)
        #    print("Original density",c*len(self.nodes),"Resulting density",(c**power)*len(self.nodes))
        #c = c**power
        #print("Resulting density",c
        #return c*len(self.nodes)
        return c

    def pruneTree(self):
        self.projectionBases = []
        self.projectionHashes = []
        EST.pruneTree(self)
    
    def pickNode(self):
        """Picks a node according to inverse density weighting strategy"""
        index = random.randint(0,len(self.projectionBases)-1)
        basis = self.projectionBases[index]
        bhash = self.projectionHashes[index]
        key = bhash.random_key()
        #key = bhash.random_key(weight=lambda k,x:1.0/len(x))
        #print("Basis",basis,"select",key,"random from points",len(bhash[key]))
        res = random.choice(bhash[key])
        if self.prune(res): return None
        #print("Selected",res.x)
        return res


class CostEdgeChecker(EdgeChecker):
    def __init__(self,edgeChecker):
        self.edgeChecker = edgeChecker
        self.costMax = None

    def feasible(self,interpolator):
        return self.edgeChecker.feasible(interpolator.components[0]) and (self.costMax is None or interpolator.components[1].end()[0] <= self.costMax)


class CostMetric:
    def __init__(self,metric,costWeight=1.0):
        self.metric = metric
        self.costWeight = costWeight
        self.costMax = None

    def __call__(self,a,b):
        #a is the existing node in the tree, b is the new sample
        d = self.metric(a[:-1],b[:-1])
        if self.costWeight is not None:
            if (self.costMax is not None) and (a[-1] > self.costMax or b[-1] > self.costMax):
                return infty
            d += max(a[-1]-b[-1],0.0)*self.costWeight
            #d += abs(a[-1]-b[-1])*self.costWeight
        return d


class CostSpaceSampler(Sampler):
    def __init__(self,baseSampler,costMax):
        self.baseSampler = baseSampler
        self.costMax = costMax

    def sample(self):
        xb = self.baseSampler.sample()
        cmax = 0.0 if self.costMax is None else self.costMax
        return xb+[random.uniform(0.0,cmax)]


class CostGoal(Set):
    def __init__(self,baseGoal,objective,costMax):
        self.baseGoal = baseGoal
        self.objective = objective
        self.costMax = costMax

    def contains(self,x):
        if not self.baseGoal.contains(x[:-1]):
            return False
        if self.costMax==None:
            return True
        return x[-1]+self.objective.terminal(x[:-1])<=self.costMax

    def sample(self):
        xb = self.baseGoal.sample()
        if xb is None: return None
        cmax = 0.0 if self.costMax is None else self.costMax-self.objective.terminal(xb)
        if cmax < 0.0: cmax = 0.0
        return xb+[random.uniform(0,cmax)]

    def project(self,x):
        return self.baseGoal.project(x[:-1])+[min(self.costMax,x[-1])]


class HeuristicCostSpaceSampler(Sampler):
    """Given optionally a cost-to-come heuristic and a cost-to-goal heuristic,
    samples only from the range of cost values that could yield a feasible
    path.
    """
    def __init__(self,baseSampler,heuristicCostToCome,heuristicCostToGo,costMax):
        self.baseSampler = baseSampler
        self.heuristicCostToCome = (heuristicCostToCome if heuristicCostToCome!=None else lambda x:0.0)
        self.heuristicCostToGo = (heuristicCostToGo if heuristicCostToGo!=None else lambda x:0.0)
        self.costMax = costMax

    def sample(self):
        count = 0
        while True:
            xb = self.baseSampler.sample()
            if self.costMax==None:
                return xb + [0.0]
            #putting a lower bound seems to be problematic for poorly scaled
            #metrics
            #cmin = self.heuristicCostToCome(xb)
            cmin = 0.0
            cmax = self.costMax - self.heuristicCostToGo(xb)
            #cmax = self.costMax
            if cmin <= cmax:
                return xb+[random.uniform(cmin,cmax)]
            count += 1
            if count % 1000 == 0:
                print("Heuristic sampler: spinning over 1000 iterations?")


class HeuristicCostSpacePruner:
    """Prunes a node if cost-from-start + heuristicCostToGo > costMax"""
    def __init__(self,heuristicCostToGo,costMax):
        self.heuristicCostToGo = (heuristicCostToGo if heuristicCostToGo!=None else lambda x:0.0)
        self.costMax = costMax

    def __call__(self,x):
        if isinstance(x,Node):
            return self(x.x)
        if self.costMax is None:
            return False
        cost = x[-1]
        xbase = x[:-1]
        return cost+self.heuristicCostToGo(xbase) > self.costMax


class CostSpaceRRT:
    """The cost-space Rapidly-exploring Random Tree planner.
    """
    def __init__(self,controlSpace,objective,metric,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        self.objective = objective
        self.baseSpace = controlSpace.configurationSpace()
        self.baseControlSpace = controlSpace
        self.costSpace = CostControlSpace(controlSpace,objective)
        self.baseMetric = metric
        self.metric = CostMetric(self.baseMetric,0)
        self.edgeChecker = CostEdgeChecker(edgeChecker)
        #self.costWeight = popdefault(params,'costWeight','adaptive')
        self.costWeight = popdefault(params,'costWeight',1)
        self.rrt = RRT(self.costSpace,self.metric,self.edgeChecker,**params)
        self.costSpaceSampler = CostSpaceSampler(Sampler(self.baseSpace),None)
        self.rrt.setConfigurationSampler(self.costSpaceSampler)
        self.bestPath = None
        self.bestPathCost = None
        self.lastPruneCost = None
        self.stats = Profiler()
        self.stats.items['rrt'] = self.rrt.stats
        self.numIters = self.stats.count('numIters')
                
    def destroy(self):
        """To be nice to the GC, call this to free up memory after you're
        done planning"""
        self.rrt.destroy()

    def setBoundaryConditions(self,x0,goal):
        """Initializes the tree from a start state x0 and a goal
        ConfigurationSubset.
        
        goal can be set to None to just explore.
        """
        if isinstance(goal,(list,tuple)):
            goal = SingletonSubset(self.baseSpace,goal)
        self.baseStart = x0
        self.baseGoal = goal
        self.costGoal = CostGoal(goal,self.objective,self.bestPathCost)
        self.rrt.setBoundaryConditions(self.costSpace.makeState(x0,0.0),self.costGoal)

    def setHeuristic(self,heuristicCostToCome=None,heuristicCostToGo=None):
        self.costSpaceSampler = HeuristicCostSpaceSampler(Sampler(self.baseSpace),heuristicCostToCome,heuristicCostToGo,self.bestPathCost)
        self.rrt.setConfigurationSampler(self.costSpaceSampler)
        self.rrt.pruner = HeuristicCostSpacePruner(heuristicCostToGo,self.bestPathCost)

    def setConfigurationSampler(self,sampler):
        self.rrt.setConfigurationSampler(CostSpaceSampler(sampler,self.bestPathCost))
    
    def setControlSelector(self,selector):
        self.rrt.setControlSelector(selector)
    
    def reset(self):
        """Resets all planning effort"""
        self.rrt.reset()
        self.bestPath = None
        self.bestPathCost = None
        self.lastPruneCost = None
        self.updateBestCost()
    
    def updateBestCost(self):
        print("Updating best path cost to",self.bestPathCost)
        self.costGoal.costMax = self.bestPathCost
        self.costSpaceSampler.costMax = self.bestPathCost
        self.metric.costMax = self.bestPathCost
        self.edgeChecker.costMax = self.bestPathCost
        if self.rrt.pruner:
            self.rrt.pruner.costMax = self.bestPathCost
        self.costSpace.setCostMax(self.bestPathCost)
        if self.bestPathCost is not None:
            if self.costWeight == 'adaptive':
                n = self.rrt.goalNodes[-1]
                dgoal = self.baseMetric(n.x[:-1],self.baseStart)
                self.metric.costWeight = self.bestPathCost / (dgoal)
                print("Setting RRT distance weight on cost to",self.metric.costWeight)
            else:
                self.metric.costWeight = self.costWeight
        else:
            self.metric.costWeight = 0.0
    
    def planMore(self,iters):
        didreset = False
        foundNewPath = False
        for n in range(iters):
            self.numIters.add(1)
            if self.rrt.planMore(1):
                #check to see if RRT has a better path
                n = self.rrt.goalNodes[-1]
                c = n.x[-1] + self.objective.terminal(n.x[:-1])
                if self.bestPathCost is None or c < self.bestPathCost:
                    print("Improved best path cost from",self.bestPathCost,"to",c)
                    self.bestPathCost = c
                    self.bestPath = self.rrt.getPath(n)
                    self.updateBestCost()
                    foundNewPath = True
                    #resets may help performance... but experiments suggest
                    #that there's little effect (5/6/2015)
                    #self.rrt.reset()
                    #didreset = True
        if foundNewPath and not didreset:
            assert self.bestPathCost is not None
            #print("Trying pruning...")
            self.lastPruneCost = self.bestPathCost
            prunecount = 0
            for n in self.rrt.nodes:
                if n.x[-1] > self.bestPathCost or self.rrt.prune(n):
                    prunecount += 1
            print("Can prune",prunecount,"of",len(self.rrt.nodes),"nodes")
            if prunecount > len(self.rrt.nodes)/5:
                #if prunecount > 0:
                oldpruner = self.rrt.pruner
                if self.rrt.pruner is None:
                    self.rrt.pruner = (lambda n:n.x[-1] >= self.bestPathCost)
                    self.rrt.pruneTree()
                    self.rrt.pruner = oldpruner
                    print("   pruned down to",len(self.rrt.nodes),"nodes")
        return self.bestPathCost
        
    def getPathCost(self):
        return self.bestPathCost
        
    def getPath(self):
        """Returns ([x0,...,xn],[u1,...,un]) for the base space."""
        if self.bestPath is None:
            return None
        x,u = self.bestPath
        return ([xi[:-1] for xi in x],u)
        
    def getRoadmap(self):
        """Returns a roadmap for the base space"""
        (V,E) = self.rrt.getRoadmap()
        return ([x[:-1] for x in V],E)
        
    def getBestPath(self,obj,goal=None):
        if obj is self.objective and goal is self.baseGoal and self.bestPath is not None:
            return self.getPath()
        obj2 = _CostSpaceObjectiveAdapter(obj)
        goal2 = None if goal is None else CostGoal(goal,obj,None)
        xs,us,cost = self.rrt.getBestPath(obj2,goal2)
        return [x[:-1] for x in xs],us,cost


class CostSpaceEST:
    """The cost-space Expansive Space Tree planner.
    """
    def __init__(self,controlSpace,objective,edgeChecker,**params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        self.objective = objective
        self.baseSpace = controlSpace.configurationSpace()
        self.baseControlSpace = controlSpace
        self.costSpace = CostControlSpace(controlSpace,objective)
        self.edgeChecker = CostEdgeChecker(edgeChecker)
        self.est = ESTWithProjections(self.costSpace,self.edgeChecker,**params)
        self.bestPath = None
        self.bestPathCost = None
        self.stats = Profiler()
        self.stats.items['est'] = self.est.stats
        self.numIters = self.stats.count('numIters')
        self.lastPruneCost = None

    def destroy(self):
        """To be nice to the GC, call this to free up memory after you're
        done planning"""
        self.est.destroy()

    def setBoundaryConditions(self,x0,goal):
        """Initializes the tree from a start state x0 and a goal
        ConfigurationSubset.
        
        goal can be set to None to just explore.
        """
        if isinstance(goal,(list,tuple)):
            goal = SingletonSubset(self.baseSpace,goal)
        self.baseStart = x0
        self.baseGoal = goal
        self.costGoal = CostGoal(goal,self.objective,self.bestPathCost)
        self.est.generateDefaultBases(list(range(len(x0))))
        self.est.setBoundaryConditions(self.costSpace.makeState(x0,0.0),self.costGoal)

    def setHeuristic(self,heuristicCostToCome=None,heuristicCostToGo=None):
        self.est.pruner = HeuristicCostSpacePruner(heuristicCostToGo,self.bestPathCost)

    def reset(self):
        print("CostSpaceEST reset")
        self.est.reset()
        print("Regenerating projection bases without cost dimension.")
        self.est.generateDefaultBases(list(range(len(self.baseStart))))
        self.numIters.set(0)
        self.bestPath = None
        self.bestPathCost = None
        self.lastPruneCost = None
        self.updateBestCost()
    
    def updateBestCost(self):
        self.costSpace.setCostMax(self.bestPathCost)
        if self.edgeChecker.costMax is None and self.bestPathCost is not None:
            print("Regenerating projection bases to include cost dimension.")
            self.est.generateDefaultBases(list(range(len(self.baseStart)+1)))
        self.costGoal.costMax = self.bestPathCost
        self.edgeChecker.costMax = self.bestPathCost
        if self.est.pruner:
            self.est.pruner.costMax = self.bestPathCost
            
    def planMore(self,iters):
        foundNewPath = False
        for n in range(iters):
            self.numIters.add(1)
            if self.est.planMore(1):
                #check to see if RRT has a better path
                n = self.est.goalNodes[-1]
                c = n.x[-1] + self.objective.terminal(n.x[:-1])
                if self.bestPathCost is None or c < self.bestPathCost:
                    print("Improved best path cost from",self.bestPathCost,"to",c)
                    foundNewPath = True
                    self.bestPathCost = c
                    self.bestPath = self.est.getPath(n)
                    self.updateBestCost()
                    #Resets seem to really hurt performance
                    #self.est.reset()

        if foundNewPath:
            #print("Trying pruning...")
            self.lastPruneCost = self.bestPathCost
            prunecount = 0
            for n in self.est.nodes:
                assert n.x[-1] is not None
                if n.x[-1] > self.bestPathCost or self.est.prune(n):
                    prunecount += 1
            #print(len(self.est.nodes),"nodes, can prune",prunecount)
            #redo tree when can prune 20% of nodes
            if prunecount >  len(self.est.nodes)/5:
                oldpruner = self.est.pruner
                if self.est.pruner is None:
                    assert self.bestPathCost is not None
                    self.est.pruner = (lambda n:n.x[-1] >= self.bestPathCost)
                    self.est.pruneTree()
                self.est.pruner = oldpruner
            #print("   pruned down to",len(self.est.nodes),"nodes")
            #else:
                #print(len(self.est.nodes),"nodes")
        return self.bestPathCost
        
    def getPathCost(self):
        return self.bestPathCost
        
    def getPath(self):
        """Returns ([x0,...,xn],[u1,...,un]) for the base space."""
        if self.bestPath is None:
            return None
        x,u = self.bestPath
        return ([xi[:-1] for xi in x],u)

    def getRoadmap(self):
        """Returns a roadmap for the base space"""
        (V,E) = self.est.getRoadmap()
        pruner = (lambda x:False)
        if self.est.pruner is not None: pruner = self.est.pruner
        return ([x[:-1] for x in V],[e for e in E if (not pruner(V[e[0]]) and not pruner(V[e[1]]))])

    def getBestPath(self,obj,goal=None):
        if obj is self.objective and goal is self.baseGoal and self.bestPath is not None:
            return self.getPath()
        obj2 = _CostSpaceObjectiveAdapter(obj)
        goal2 = None if goal is None else CostGoal(goal,obj,None)
        xs,us,cost = self.est.getBestPath(obj2,goal2)
        return [x[:-1] for x in xs],us,cost


class RepeatedRRT(RRT):
    """The repeated Rapidly-exploring Random Tree planner.
    """
    def __init__(self,controlSpace,objective,metric,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        RRT.__init__(self,controlSpace,metric,edgeChecker,**params)
        self.objective = objective
        self.bestPath = None
        self.bestPathCost = None
        self.doprune = False
        
    def reset(self):
        """Resets all planning effort"""
        RRT.reset(self)
        self.bestPath = None
        self.bestPathCost = None
        
    def setBoundaryConditions(self,x0,goal):
        """Sets the start and goal"""
        RRT.setBoundaryConditions(self,x0,goal)
        #add cost term to root node
        self.root.c = 0
        
    def planMore(self,iters):
        foundNewPath = False
        for n in range(iters):
            self.numIters += 1
            n = RRT.expand(self)
            if n is None:
                continue
            if not hasattr(n,'c'):
                n.c = n.parent.c + self.objective.incremental(n.parent.x,n.uparent)
            if self.goal is not None and self.goal.contains(n.x):
                totalcost = n.c + self.objective.terminal(n.x)
                if self.bestPath is None or totalcost < self.bestPathCost:
                    print("RRT found path with new cost",totalcost)
                    foundNewPath = True
                    self.bestPathCost = totalcost
                    self.bestPath = RRT.getPath(self,n)
                    RRT.reset(self)
                    return True
                if not self.doprune:
                    #rrt found a path and we're not doing pruning... just reset the RRT
                    RRT.reset(self)
                    return False
        if self.doprune and foundNewPath:
            prunecount = 0
            for n in self.nodes:
                if n.c > self.bestPathCost or self.prune(n):
                    prunecount += 1
            if prunecount > len(self.nodes)/5 and prunecount > 100:
                #print("Can prune",prunecount,"of",len(self.nodes),"nodes")
                oldpruner = self.pruner
                if self.pruner is None:
                    self.pruner = (lambda n:n.c >= self.bestPathCost)
                    self.pruneTree()
                self.pruner = oldpruner
            #print("   pruned down to",len(self.nodes),"nodes")
        return False
        
    def prune(self,n):
        if not self.doprune or self.bestPathCost is None: return False
        if not hasattr(n,'c'):
            n.c = n.parent.c + self.objective.incremental(n.parent.x,n.uparent)
        if n.c > self.bestPathCost:
            return True
        if self.pruner:
            return self.pruner(node)
        return False
        
    def getPathCost(self):
        return self.bestPathCost
        
    def getPath(self,n=None):
        """Returns ([x0,...,xn],[u1,...,un]) for the base space."""
        if n is not None:
            return RRT.getPath(self,n)
        if self.bestPath is None:
            return None
        return self.bestPath

    def getBestPath(self,obj,goal=None):
        if obj is self.objective and goal is self.goal and self.bestPath is not None:
            return self.getPath()
        return RRT.getBestPath(self,obj,goal)


class AnytimeRRT(RRT):
    """The Anytime Rapidly-exploring Random Tree planner (Ferguson and Stentz,
    06).
    """
    def __init__(self,controlSpace,objective,metric,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        #amount that the cost/nearness weight gets shifted each time a path
        #is found
        self.weightIncrement = popdefault(params,'weightIncrement',0.1)
        #amount required for path cost shrinkage
        self.epsilon = popdefault(params,'epsilon',0.01)
        #must use brute force picking
        #commented out because pickNode does this already
        #params['nearestNeighborMethod'] = 'bruteforce'  
        RRT.__init__(self,controlSpace,metric,edgeChecker,**params)
        self.objective = objective
        self.bestPath = None
        self.bestPathCost = None
        self.distanceWeight = 1
        self.costWeight = 0

    def pickNode(self,xrand):
        """Picks a node closest to xrand."""
        nnear = None    
        dbest = infty
        for n in self.nodes:
            d = self.distanceWeight*self.metric(n.x,xrand) + self.costWeight*n.c
            if d < dbest and not self.prune(n):
                nnear = n
                dbest = d
        return nnear
        
    def reset(self):
        """Resets all planning effort"""
        RRT.reset(self)
        self.bestPath = None
        self.bestPathCost = None
        self.distanceWeight = 1
        self.costWeight = 0
        
    def setBoundaryConditions(self,x0,goal):
        """Sets the start and goal"""
        RRT.setBoundaryConditions(self,x0,goal)
        #add cost term to root node
        self.root.c = 0
        
    def planMore(self,iters):
        foundNewPath = False
        for n in range(iters):
            self.numIters += 1
            n = RRT.expand(self)
            if n is None: continue
            if not hasattr(n,'c'):
                n.c = n.parent.c + self.objective.incremental(n.parent.x,n.uparent)
            if self.goal is not None and self.goal.contains(n.x):
                totalcost = n.c + self.objective.terminal(n.x)
                if self.bestPath is None or totalcost < self.bestPathCost*(1.0-self.epsilon):
                    print("Anytime-RRT found path with new cost",totalcost)
                    self.distanceWeight -= self.weightIncrement
                    self.distanceWeight = max(self.distanceWeight,0)
                    self.costWeight += self.weightIncrement
                    self.costWeight = min(self.costWeight,1)
                    foundNewPath = True
                    self.bestPathCost = totalcost
                    self.bestPath = RRT.getPath(self,n)
                    RRT.reset(self)
                    return True
        return False
        
    def prune(self,n):
        if self.bestPathCost is None: return False
        if not hasattr(n,'c'):
            n.c = n.parent.c + self.objective.incremental(n.parent.x,n.uparent)
        if n.c > self.bestPathCost*(1.0-self.epsilon):
            return True
        if self.pruner:
            return self.pruner(node)
        return False
        
    def getPathCost(self):
        return self.bestPathCost
        
    def getPath(self,n=None):
        """Returns ([x0,...,xn],[u1,...,un]) for the base space."""
        if n is not None:
            return RRT.getPath(self,n)
        if self.bestPath is None:
            return None
        return self.bestPath

    def getBestPath(self,obj,goal=None):
        if obj is self.objective and goal is self.goal and self.bestPath is not None:
            return self.getPath()
        return RRT.getBestPath(self,obj,goal)


class RepeatedEST(ESTWithProjections):
    """The repeated Expansive Space Tree planner.
    """
    def __init__(self,controlSpace,objective,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        ESTWithProjections.__init__(self,controlSpace,edgeChecker,**params)
        self.objective = objective
        self.bestPath = None
        self.bestPathCost = None
        self.doprune = False

    def reset(self):
        """Resets all planning effort"""
        ESTWithProjections.reset(self)
        self.bestPath = None
        self.bestPathCost = None

    def setBoundaryConditions(self,x0,goal):
        """Sets the start and goal"""
        ESTWithProjections.setBoundaryConditions(self,x0,goal)
        #add cost term to root node
        self.root.c = 0

    def planMore(self,iters):
        foundNewPath = False
        for n in range(iters):
            self.numIters += 1
            n = ESTWithProjections.expand(self)
            if n is None: continue
            if not hasattr(n,'c'):
                n.c = n.parent.c + self.objective.incremental(n.parent.x,n.uparent)
            if self.goal is not None and self.goal.contains(n.x):
                totalcost = n.c + self.objective.terminal(n.x)
                if self.bestPath is None or totalcost < self.bestPathCost:
                    print("EST found path with new cost",totalcost)
                    foundNewPath = True
                    self.bestPathCost = totalcost
                    self.bestPath = ESTWithProjections.getPath(self,n)
                    ESTWithProjections.reset(self)
                    return True
                if not self.doprune:
                    ESTWithProjections.reset(self)
                    return False
        if self.doprune and foundNewPath:
            foundNewPath = False
            prunecount = 0
            for n in self.nodes:
                if n.c > self.bestPathCost or self.prune(n):
                    prunecount += 1
            if prunecount > len(self.nodes)/5 and prunecount > 100:
                #print("Can prune",prunecount,"of",len(self.nodes),"nodes")
                oldpruner = self.pruner
                if self.pruner is None:
                    self.pruner = (lambda n:n.c >= self.bestPathCost)
                    self.pruneTree()
                self.pruner = oldpruner
            #print("   pruned down to",len(self.nodes),"nodes")
        return False
        
    def pruneExtension(self,n,u):
        if not self.doprune or self.bestPathCost is None: return False
        if n.c + self.objective.incremental(n.x,u) > self.bestPathCost:
            return True
        return False
        
    def prune(self,n):
        if not self.doprune or self.bestPathCost is None: return False
        if not hasattr(n,'c'):
            n.c = n.parent.c + self.objective.incremental(n.parent.x,n.uparent)
        if n.c > self.bestPathCost:
            return True
        if self.pruner:
            return self.pruner(node)
        return False
        
    def getPathCost(self):
        return self.bestPathCost
        
    def getPath(self,n=None):
        """Returns ([x0,...,xn],[u1,...,un]) for the base space."""
        if n is not None:
            return ESTWithProjections.getPath(self,n)
        if self.bestPath is None:
            return None
        return self.bestPath

    def getBestPath(self,obj,goal=None):
        if obj is self.objective and goal is self.goal and self.bestPath is not None:
            return self.getPath()
        return ESTWithProjections.getBestPath(self,obj,goal)


class _CostSpaceObjectiveAdapter(ObjectiveFunction):
    def __init__(self,base):
        self.base = base
    def __str__(self):
        return str(self.base)+"CostSpace adapter"
    def incremental(self,x,u=None):
        return self.base.incremental(x[:-1],u)
    def terminal(self,x):
        return self.base.terminal(x[:-1])
    
    def incremental_gradient(self,x,u):
        #TODO
        return self.base.incremental_gradient(x[:-1],u)
    def incremental_hessian(self,x,u):
        #TODO
        return self.base.incremental_hessian(x[:-1],u)
    def terminal_gradient(self,x):
        #TODO
        return self.base.terminal_gradient(x[:-1])
    def terminal_hessian(self,x):
        #TODO
        return self.base.terminal_hessian(x[:-1])