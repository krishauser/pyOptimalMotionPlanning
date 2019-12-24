from __future__ import print_function,division
from six import iteritems
from ..spaces.metric import *
from ..spaces.controlspace import *
from ..spaces.edgechecker import *
from .kinodynamicplanner import *
from .rrtstarplanner import *

all_planners = ['ao-est','ao-rrt','r-est','r-est-prune','r-rrt','r-rrt-prune','rrt*','anytime-rrt','stable-sparse-rrt','sst*']
rrt_planners = ['ao-rrt','anytime-rrt','r-rrt','r-rrt-prune','stable-sparse-rrt','sst*']
est_planners = ['ao-est','r-est','r-est-prune']

filename = {'ao-rrt':'ao_rrt',
            'ao-est':'ao_est',
            'rrt*':'rrtstar',
            'r-rrt':'repeated_rrt',
            'r-rrt-prune':'repeated_rrt_prune',
            'r-est':'repeated_est',
            'r-est-prune':'repeated_est_prune',
            'rrt':'rrt',
            'est':'est',
            'anytime-rrt':'anytime_rrt',
            'stable-sparse-rrt':'stable_sparse_rrt',
            'sst*':'stable_sparse_rrt_star'}

kinematicPlanners = set(['rrt*'])
optimalPlanners = set(['ao-rrt','ao-est','rrt*','r-rrt','r-rrt-prune','r-est','r-est-prune','anytime-rrt','stable-sparse-rrt','sst*'])

def makePlanner(type,space,start,goal,
                objective=None,
                checker=None,
                metric=euclideanMetric,
                heuristic=None,
                costLowerBound=None,
                **params):
    if isinstance(space,ControlSpace):
        if type in kinematicPlanners:
            raise ValueError("Cannot do kinodynamic planning with planner "+type)
        controlSpace = space
        space = space.configurationSpace()
    else:
        if type not in kinematicPlanners:
            controlSpace = ControlSpaceAdaptor(space)
            controlSpace.nextStateSamplingRange = popdefault(params,'nextStateSamplingRange',0.1,'makePlanner(): Warning, control space sampling range not provided, using 0.1')

    if isinstance(goal,(list,tuple)) or isinstance(goal,SingletonSet):
        raise RuntimeError("Cannot yet handle singleton goals")

    if heuristic==None and costLowerBound != None:
        print("Constructing default heuristic from cost lower bound")
        heuristicCostToCome = lambda x:costLowerBound(start,x)
        if goal.project(start) == None:
            print("  No cost-to-go heuristic.")
            heuristicCostToGo = None
        else:
            def h(x):
                xproj = goal.project(x)
                if xproj == None: return 0
                return costLowerBound(x,xproj)
        heuristicCostToGo = h
        heuristic = (heuristicCostToCome,heuristicCostToGo)
    else:
        if not isinstance(heuristic,(list,tuple)):
            heuristic = (None,heuristic)
        
    if checker==None:
        edgeCheckTolerance = popdefault(params,'edgeCheckTolerance',0.01,"makePlanner: Warning, edge checker and/or edge checking tolerance not specified, using default tolerance 0.01")
        checker = EpsilonEdgeChecker(space,edgeCheckTolerance)

    if type == 'rrt*':
        planner = RRTStar(space,metric,checker,**params)
    elif type == 'ao-rrt':
        planner = CostSpaceRRT(controlSpace,objective,metric,checker,**params)
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicCostControlSelector(planner.costSpace,controlSpace.nextStateSamplingRange))
    elif type == 'ao-est':
        planner = CostSpaceEST(controlSpace,objective,checker,**params)
    elif type =='rrt':
        planner = RRT(controlSpace,metric,checker)
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicControlSelector(planner.controlSpace,controlSpace.nextStateSamplingRange))
    elif type == 'est':
        planner = ESTWithProjections(controlSpace,checker,**params)
        #planner = EST(controlSpace,checker,**params)
    elif type == 'r-est':
        planner = RepeatedEST(controlSpace,objective,checker,**params)
    elif type == 'r-est-prune':
        planner = RepeatedEST(controlSpace,objective,checker,**params)
        planner.doprune = True
    elif type == 'anytime-rrt':
        planner = AnytimeRRT(controlSpace,objective,metric,checker,**params)
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicControlSelector(controlSpace,controlSpace.nextStateSamplingRange))
    elif type == 'r-rrt':
        planner = RepeatedRRT(controlSpace,objective,metric,checker,**params)
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicControlSelector(controlSpace,controlSpace.nextStateSamplingRange))
    elif type == 'r-rrt-prune':
        planner = RepeatedRRT(controlSpace,objective,metric,checker,**params)
        planner.doprune = True
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicControlSelector(controlSpace,controlSpace.nextStateSamplingRange))
    elif type == 'stable-sparse-rrt':
        planner = StableSparseRRT(controlSpace,objective,metric,checker,**params)
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicControlSelector(controlSpace,controlSpace.nextStateSamplingRange))
    elif type == 'sst*':
        planner = StableSparseRRTStar(controlSpace,objective,metric,checker,**params)
        #set direct steering functions for kinematic spaces 
        if isinstance(controlSpace,ControlSpaceAdaptor):
            planner.setControlSelector(KinematicControlSelector(controlSpace,controlSpace.nextStateSamplingRange))
    else:
        raise RuntimeError("Invalid planner type "+type)
    planner.setBoundaryConditions(start,goal)
    if type.startswith=='ao' and heuristic != None:
        planner.setHeuristic(*heuristic)
    return planner

