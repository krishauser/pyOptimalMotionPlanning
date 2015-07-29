from ..spaces.configurationspace import *
from ..spaces.controlspace import *
from ..spaces.edgechecker import *
from ..spaces.sampler import *
from ..spaces import metric
from ..structures.nearestneighbors import *
from ..structures import kdtree
from kinodynamicplanner import Profiler,TreePlanner,RandomControlSelector,popdefault

infty = float('inf')

class RRTStar(TreePlanner):
    """The RRT* planner with a kinematic space

    Stores a tree of Nodes.  Expands the tree at random using the RRT strategy
    with a goal bias of probability pChooseGoal.
    """
    def __init__(self,cspace,metric,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        TreePlanner.__init__(self)
	if not isinstance(cspace,ConfigurationSpace):
		print "Warning, cspace is not a ConfigurationSpace"
	if not isinstance(edgeChecker,EdgeChecker):
		print "Warning, edgeChecker is not an EdgeChecker"
        self.cspace = cspace
        self.metric = metric
        self.edgeChecker = edgeChecker
        self.goal = None
        self.goalSampler = None
        self.pChooseGoal = popdefault(params,'pChooseGoal',0.1)
        self.goalNodes = []
        self.configurationSampler = Sampler(self.cspace)
        nnmethod = popdefault(params,'nearestNeighborMethod','kdtree')
        self.nearestNeighbors = NearestNeighbors(self.metric,nnmethod)
        self.bestPathCost = infty
        self.bestPath = None
        if len(params) != 0:
            print "Warning, unused params",params
        self.stats = Profiler()
        self.numIters = self.stats.count('numIters')
    def destroy(self):
        TreePlanner.destroy(self)
        self.nearestNeighbors.reset()
        self.goalNodes = []
    def reset(self):
        """Re-initializes the RRT* to the same start / goal, clears the
        planning tree."""
	x0 = self.root.x
	goal = self.goal
        self.bestPathCost = infty
        self.bestPath = None
	self.destroy()
	self.setBoundaryConditions(x0,goal)
	self.numIters.set(0)
	
    def setBoundaryConditions(self,x0,goal):
        """Initializes the tree from a start state x0 and a goal
        ConfigurationSubset.
        
        goal can be set to None to just explore.
        """
        self.setRoot(x0)
        self.root.c = 0
        self.goal = goal
        if goal != None:
            if isinstance(goal,(list,tuple)):
                self.goal = SingletonSubset(self.cspace,goal)
            self.goalSampler = SubsetSampler(self.cspace,self.goal)
        self.nearestNeighbors.add(x0,self.root)
    def setConfigurationSampler(self,sampler):
        self.configurationSampler = sampler
    def planMore(self,iters):
        for n in xrange(iters):
            self.numIters.add(1)
            n = self.expand()
            if n != None and self.goal != None:
                if self.goal.contains(n.x):
                    self.goalNodes.append(n)
                    if n.c < self.bestPathCost:
                        self.bestPathCost = n.c
                        self.bestPath = self.getPath(n)
                        return True
        return False
    def expand(self):
        """Expands the tree via the RRT* technique.  Returns the new node
        or None otherwise."""
        if self.goalSampler and random.uniform(0.0,1.0) < self.pChooseGoal:
            xrand = self.goalSampler.sample()
        else:
            xrand = self.configurationSampler.sample()
        nnear = self.pickNode(xrand)
        if nnear == None:
            return None
        if nnear.c > self.bestPathCost:
            return None
        #do rrt* connection
        N = float(self.numIters.count+1)
        rad = 1.414*(math.log(N)/N) ** (1.0/len(nnear.x));
        k = int(((1.0+1.0/len(xrand))*math.e)*math.log(N))
        if k <=0: k=1
        u = min(1.0,rad / self.metric(nnear.x,xrand))
        xend = self.cspace.interpolate(nnear.x,xrand,u)
        edge = self.cspace.interpolator(nnear.x,xend)
        if not self.edgeChecker.feasible(edge):
            return None
        #feasible edge, add it
        nnew = self.addEdge(nnear,xend,edge)
        nnew.c = nnear.c + edge.length()
        self.nearestNeighbors.add(nnew.x,nnew)
        self.rewire(nnew,k=k,rad=None,first=True,recursive=False)
        return nnew
    
    def rewire(self,n,k=None,rad=None,first=True,recursive=False):
        #gather incoming cost from parent
        changed = False
        if n.parent != None:
            d = n.parent.c + self.metric(n.x,n.parent.x)
            if d != n.c:
                #print "initial cost improvement",n.c,"to",d
                n.c = d
                changed = True
        if changed:
            #self.stats.count('numCostImprovements').add(1)
            pass

        if first or recursive:
            if rad != None:
                neighbors = self.nearestNeighbors.neighbors(n.x,rad)
            elif k != None:
                neighbors = self.nearestNeighbors.knearest(n.x,k)
            else:
                raise ValueError("Rewire: must have either k or rad arguments provided")
            #sort the neighbors by accumulated cost
            dneighbors = []
            #print "Node",n.x,"cost",n.c
            for x,nn in neighbors:
                if nn == n: continue
                if nn == n.parent: continue
                d = self.metric(n.x,nn.x) + nn.c
                #print nn.x,d,self.metric(n.x,nn.x)
                dneighbors.append((d,nn))
            dneighbors = sorted(dneighbors,key=lambda x:x[0])
            #print len(dneighbors),"neighbors, cost range",dneighbors[0][0],"to",dneighbors[-1][0]
            #add incoming edges from neighbors
            for (d,nn) in dneighbors:
                if d > n.c: break
                edge = self.cspace.interpolator(nn.x,n.x)
                if self.edgeChecker.feasible(edge):
                    #print "rrt* cost improvement",n.c,"to",d
                    n.setParent(nn,n.x,edge)
                    n.c = d
                    changed = True
                    #optimization: break on first added neighbor, since they're
                    #sorted by increasing cost
                    if changed:
                        #self.stats.count('numIncomingRewirings').add(1)
                        pass
                    break
        if n.c > self.bestPathCost:
            return
        
        if first or changed:
            for c in n.children:
                self.rewire(c,k=k,rad=rad,first=False,recursive=recursive)

        if first or recursive:
            #add outgoing edges from n
            for (d,nn) in dneighbors:
                newcost = n.c + self.metric(n.x,nn.x)
                if newcost < nn.c:
                    #print "Outgoing improvement from",nn.c,"to",newcost
                    #potential improvement, check edge and rewire if necessary
                    edge = self.cspace.interpolator(n.x,nn.x)
                    if self.edgeChecker.feasible(edge):
                        #self.stats.count('numOutgoingRewirings').add(1)
                        nn.setParent(n,nn.x,edge)
                        #nn.c = newcost
                        self.rewire(nn,k=k,rad=rad,first=False,recursive=recursive)
        
    def pickNode(self,xrand):
        """Picks a node closest to xrand.  If dynamicDomain is True,
        uses the radius associated with the node"""
        res = self.nearestNeighbors.nearest(xrand)
        if res==None: return None
        return res[1]
    def getPath(self,n=None):
        if n == None:
            return self.bestPath
        return TreePlanner.getPath(self,n)



class StableSparseRRT(TreePlanner):
    """An implementation of Littlefield et al 2014 Stable Sparse RRT.
    
    Nodes get a cost 'c' and an active flag 'active'.

    The witness set consists of pairs [point,rep].
    """
    def __init__(self,controlSpace,objective,metric,edgeChecker,
                 **params):
        """Given a ControlSpace controlSpace, a metric, and an edge checker"""
        TreePlanner.__init__(self)
        self.controlSpace = controlSpace
	if not isinstance(controlSpace,ControlSpace):
            print "Warning, controlSpace is not a ControlSpace"
	if not isinstance(edgeChecker,EdgeChecker):
            print "Warning, edgeChecker is not an EdgeChecker"
        self.cspace = controlSpace.configurationSpace()    
        self.metric = metric
        self.objective = objective
        self.edgeChecker = edgeChecker
	self.controlSelector = RandomControlSelector(controlSpace,self.metric,1)
        self.goal = None
        self.goalSampler = None
        self.pChooseGoal = popdefault(params,'pChooseGoal',0.1)
        self.goalNodes = []
        self.selectionRadius = popdefault(params,'selectionRadius',0.1)
        self.witnessRadius = popdefault(params,'witnessRadius',0.03)
        self.witnessSet = []
        self.configurationSampler = Sampler(self.controlSpace.configurationSpace())        
        nnmethod = popdefault(params,'nearestNeighborMethod','kdtree')
        self.nearestNeighbors = NearestNeighbors(self.metric,nnmethod)
        self.nearestWitness = NearestNeighbors(self.metric,nnmethod)
        self.stats = Profiler()
        self.numIters = self.stats.count('numIters')
        self.bestPathCost = infty
        self.bestPath = None
        if len(params) != 0:
            print "Warning, unused params",params

    def destroy(self):
        TreePlanner.destroy(self)
        self.goalNodes = []
        self.witnessSet = []
    
    def reset(self):
        """Re-initializes the RRT* to the same start / goal, clears the
        planning tree."""
	x0 = self.root.x
	goal = self.goal
        self.bestPathCost = infty
        self.bestPath = None
	self.destroy()
	self.setBoundaryConditions(x0,goal)
	self.numIters.set(0)
	
    def setBoundaryConditions(self,x0,goal):
        """Initializes the tree from a start state x0 and a goal
        ConfigurationSubset.
        
        goal can be set to None to just explore.
        """
        self.setRoot(x0)
        self.root.c = 0
        self.root.active = True
        self.witnessSet = [[x0,self.root]]
        self.goal = goal
        if goal != None:
            if isinstance(goal,(list,tuple)):
                self.goal = SingletonSubset(self.cspace,goal)
            self.goalSampler = SubsetSampler(self.cspace,self.goal)
        self.nearestNeighbors.reset()
        self.nearestNeighbors.add(x0,self.root)
        self.nearestWitness.reset()
        self.nearestWitness.add(x0,self.witnessSet[0])

    def setConfigurationSampler(self,sampler):
        self.configurationSampler = sampler
    def setControlSelector(self,selector):
        self.controlSelector = selector
    def planMore(self,iters):
        for n in xrange(iters):
            self.numIters += 1
            n = self.expand()
            if n != None and self.goal != None:
                if self.goal.contains(n.x):
                    self.goalNodes.append(n)
                    if n.c + self.objective.terminal(n.x) < self.bestPathCost:
                        self.bestPathCost = n.c + self.objective.terminal(n.x)
                        print "New goal node with cost",self.bestPathCost
                        self.bestPath = TreePlanner.getPath(self,n)
                        if self.bestPath == None:
                            print "Uh... no path to goal?"
                        return True
        return False
    def expand(self):
        """Expands the tree via the Sparse-Stable-RRT technique.
        Returns the new node  or None otherwise."""
        if self.goalSampler and random.uniform(0.0,1.0) < self.pChooseGoal:
            xrand = self.goalSampler.sample()
        else:
            xrand = self.configurationSampler.sample()
        #self.stats.stopwatch('pickNode').begin()
        nnear = self.pickNode(xrand)
        #self.stats.stopwatch('pickNode').end()
        if nnear == None:
            return None
        #self.stats.stopwatch('selectControl').begin()
        u = self.controlSelector.select(nnear.x,xrand)
        #self.stats.stopwatch('selectControl').end()
        #self.stats.stopwatch('edgeCheck').begin()
        edge = self.controlSpace.interpolator(nnear.x,u)
        if not self.edgeChecker.feasible(edge):
            #self.stats.stopwatch('edgeCheck').end()
            #self.stats.count('numInfeasible').add(1)
            return None
        #self.stats.stopwatch('edgeCheck').end()
        newcost = nnear.c + self.objective.incremental(nnear.x,u)
        #feasible edge, add it
        nnew = self.addEdge(nnear,u,edge)
        nnew.c = newcost
        nnew.active = True
        #self.stats.stopwatch('sst-domination-check').begin()
        localbest = self.nodeLocallyBest(nnew)
        #self.stats.stopwatch('sst-domination-check').end()
        if localbest == False:
            #dominated by some other node
            #self.stats.count('numDominated').add(1)
            assert nnew == self.nodes[-1]
            nnew.destroy()
            self.nodes.pop(-1)
            return None
        self.nearestNeighbors.add(nnew.x,nnew)
        #self.doPruning(nnew,localbest[1])
        #self.stats.stopwatch('sst-domination-pruning').begin()
        for s in localbest[1]:
            self.doPruning(nnew,s)
        #self.stats.stopwatch('sst-domination-pruning').end()
        return nnew

    def nodeLocallyBest(self,n):
        """Tests the node n against nearby witnesses.  If no witnesses
        are nearby, n gets added as a witness. Algorithm 7"""
        res = self.nearestWitness.nearest(n.x)
        assert res != None
        s = res[1]
        if self.metric(s[0],n.x) > self.witnessRadius:
            #self.stats.count('numWitnesses').add(1)
            self.witnessSet.append([n.x,None])
            self.nearestWitness.add(n.x,self.witnessSet[-1])
            return True,self.witnessSet[-1]
        wlist = self.nearestWitness.neighbors(n.x,self.witnessRadius)
        better = []
        for res in wlist:
            s = res[1]
            if s[1] == None: better.append(s)
            if n.c < s[1].c: better.append(s)
        if len(better)==0: return False
        return True,better
        #if s[1] == None: return True,s
        #if n.c < s[1].c: return True,s
        return False
    
    def doPruning(self,n,snearest = None):
        """Algorithm 8"""
        if snearest != None:
            res = self.nearestWitness.nearest(n.x)
            assert res != None
            snearest = res[1]
        npeer = snearest[1]
        if npeer == n: return
        snearest[1] = n
        if npeer != None:
            #self.stats.count('numPruned').add(1)
            npeer.active = False
            #print "Remove",c
            while npeer != None and len(npeer.children)==0 and not npeer.active:
                #self.stats.count('numDeleted').add(1)
                p = npeer.parent
                #remove from NN data structure?
                cnt = self.nearestNeighbors.remove(npeer.x,npeer)
                #print "Remove",c
                npeer.destroy()
                npeer = p
    
    def pickNode(self,xrand):
        """Picks a within distance selectionRadius of xrand with the lowest
        cost.  Algorithm 6"""
        res = self.nearestNeighbors.neighbors(xrand,self.selectionRadius)
        activenear = []
        for pt,n in res:
            if n.active: activenear.append(n)
        if len(activenear) == 0:
            #return nearest of active neighbors
            res = self.nearestNeighbors.nearest(xrand,lambda pt,n:not n.active)
            if res ==None: return None
            return res[1]
        else:
            #return node with minimum distance
            cmin = infty
            res = None
            for n in activenear:
                if n.c < cmin:
                    res = n
                    cmin = n.c
            return res

    def getPath(self):
        return self.bestPath

class StableSparseRRTStar:
    def __init__(self,controlSpace,objective,metric,edgeChecker,
                 **params):
        self.selectionRadius0 = popdefault(params,'selectionRadius',0.2)
        self.witnessRadius0 = popdefault(params,'witnessRadius',0.2)
        self.numIters0 = popdefault(params,'numSSTIters',300)
        self.shrinkage = popdefault(params,'shrinkage',0.8)
        self.ssrrt = StableSparseRRT(controlSpace,objective,metric,
                                     edgeChecker,selectionRadius=self.selectionRadius0,witnessRadius=self.witnessRadius0,**params)
        self.itersleft = self.numIters0
        self.stats = Profiler()
        self.stats.items['ssrrt'] = self.ssrrt.stats
        self.numIters = self.stats.count('numIters')
        self.restartCount = self.stats.count('restartCount')
        self.bestPath = None
        self.bestPathCost = infty
        if len(params) != 0:
            print "Warning, unused params",params
    def setBoundaryConditions(self,start,goal):
        self.ssrrt.setBoundaryConditions(start,goal)
    def setConfigurationSampler(self,sampler):
        self.ssrrt.setConfigurationSampler(sampler)
    def setControlSelector(self,selector):
        self.ssrrt.setControlSelector(selector)
    def reset(self):
        self.ssrrt.reset()
        self.ssrrt.selectionRadius = self.selectionRadius0
        self.ssrrt.witnessRadius = self.witnessRadius0
        self.itersleft = self.numIters0
        self.numIters.set(0)
        self.restartCount.set(0)
        self.bestPath = None
        self.bestPathCost = infty
    def planMore(self,iters):
        res = False
        for n in xrange(iters):
            self.numIters += 1
            if self.ssrrt.planMore(1):
                res = True
                if self.ssrrt.bestPathCost < self.bestPathCost:
                    self.bestPathCost = self.ssrrt.bestPathCost
                    self.bestPath = self.ssrrt.bestPath
            self.itersleft -= 1
            if self.itersleft <= 0:
                #go to next iteration of SS-RRT
                self.restartCount += 1
                self.ssrrt.reset()
                self.ssrrt.selectionRadius *= self.shrinkage
                self.ssrrt.witnessRadius *= self.shrinkage
                #formula from Algorithm 9
                self.itersleft = (1.0+math.log(self.restartCount.count))*pow(self.shrinkage,-(len(self.ssrrt.root.x)+1)*self.restartCount.count)*self.numIters0
        return res
    def getPath(self):
        return self.bestPath
    def getRoadmap(self):
        return self.ssrrt.getRoadmap()
