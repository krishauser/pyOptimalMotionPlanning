from pomp.planners import allplanners
from pomp.planners import test
from pomp.example_problems import *
import time
import copy
import sys
import os,errno

numTrials = 10

def mkdir_p(path):
    """Quiet path making"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def testPlannerDefault(problem,problemName,maxTime,plannerType,**plannerParams):
    global numTrials
    print "Planning with",plannerType,'on problem',problemName
    planner = problem.planner(plannerType,**plannerParams)
    folder = os.path.join("data",problemName)
    mkdir_p(folder)
    test.testPlanner(planner,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'))


all_planners = ['ao-est','ao-rrt','r-est','r-est-prune','r-rrt','r-rrt-prune','rrt*','anytime-rrt','stable-sparse-rrt']
rrt_planners = ['ao-rrt','anytime-rrt','r-rrt','r-rrt-prune','stable-sparse-rrt']
est_planners = ['ao-est','r-est','r-est-prune']

all_problems = {'Kink':geometric.kinkTest(),
                'Bugtrap':geometric.bugtrapTest(),
                'Dubins':dubins.dubinsCarTest(),
                'Dubins2':dubins.dubinsTest2(),
                'Flappy':flappy.flappyTest(),
                'DoubleIntegrator':doubleintegrator.doubleIntegratorTest(),
                'Pendulum':pendulum.pendulumTest()}

defaultParameters = {'maxTime':30}
customParameters = {'Kink':{'maxTime':40,'nextStateSamplingRange':0.15},
                    'Bugtrap':{'maxTime':40,'nextStateSamplingRange':0.15},
                    'Pendulum':{'maxTime':120,'edgeCheckTolerance':0.1,'selectionRadius':.3,'witnessRadius':0.16},
                    'Flappy':{'maxTime':120,'edgeCheckTolerance':4,'selectionRadius':70,'witnessRadius':35},
                    'DoubleIntegrator':{'maxTime':60,'selectionRadius':0.3,'witnessRadius':0.3},
                    'Dubins':{'selectionRadius':0.25,'witnessRadius':0.2},
                    'Dubins2':{'selectionRadius':0.25,'witnessRadius':0.2}
                    }

def parseParameters(problem,planner):
    global defaultParameters,customParameters
    params = copy.deepcopy(defaultParameters)
    if problem in customParameters:
        params.update(customParameters[problem])
    if '(' in planner:
        #parse out key=value,... string
        name,args = planner.split('(',1)
        if args[-1] != ')':
            raise ValueError("Planner string expression must have balanced parenthesis, i.e.: func ( arglist )")
        args = args[:-1]
        args = args.split(',')
        for arg in args:
            kv = arg.split("=")
            if len(kv) != 2:
                raise ValueError("Unable to parse argument "+arg)
            try:
                params[kv[0]] = int(kv[1])
            except ValueError:
                try:
                    params[kv[0]] = float(kv[1])
                except ValueError:
                    params[kv[0]] = kv[1]
        planner = name
    return planner,params

def runTests(problems = None,planners = None):
    global all_planners,all_problems
    if planners == None or planners == 'all' or planners[0] == 'all':
        planners = all_planners

    if problems == None or problems == 'all' or problems[0] == 'all':
        problems = all_problems.keys()

    for prname in problems:
        pr = all_problems[prname]
        for p in planners:
            p,params = parseParameters(prname,p)
            maxTime = params['maxTime']
            del params['maxTime']
            if pr.differentiallyConstrained() and p in allplanners.kinematicPlanners:
                #p does not support differentially constrained problems
                continue
            testPlannerDefault(pr,prname,maxTime,p,**params)
            print "Finished test on problem",prname,"with planner",p
            print "Parameters:"
            for (k,v) in params.iteritems():
                print " ",k,":",v
    return

def runViz(problem,planner):
    #runVisualizer(rrtChallengeTest(),type=planner,nextStateSamplingRange=0.15,edgeCheckTolerance = 0.005)
    planner,params = parseParameters(problem,planner)
    if 'maxTime' in params:
        del params['maxTime']
    
    print "Planning on problem",problem,"with planner",planner
    print "Parameters:"
    for (k,v) in params.iteritems():
        print " ",k,":",v
    runVisualizer(all_problems[problem],type=planner,**params)
    
if __name__=="__main__":
    #HACK: uncomment one of these to test manually
    #runViz('Kink','rrt*')
    #test KD-tree in noneuclidean spaces
    #runViz('Pendulum','ao-rrt(numControlSamples=10,nearestNeighborMethod=bruteforce)')
    #runViz('Pendulum','ao-rrt')
    #runViz('Dubins','stable-sparse-rrt(selectionRadius=0.25,witnessRadius=0.2)')
    #runViz('DoubleIntegrator','stable-sparse-rrt(selectionRadius=0.3,witnessRadius=0.3)')
    #runViz('Pendulum','stable-sparse-rrt(selectionRadius=0.3,witnessRadius=0.16)')
    #runViz('Flappy','stable-sparse-rrt(selectionRadius=70,witnessRadius=35)')

    if len(sys.argv) < 3:
        print "Usage: main.py [-v] Problem Planner1 ... Plannerk"
        print
        print "  Problem can be one of:"
        print "   ",",\n    ".join(sorted(all_problems))
        print "  or 'all' to test all problems."
        print
        print "  Planner can be one of:"
        print "   ",",\n    ".join(sorted(all_planners))
        print "  or 'all' to test all planners."
        print
        print "  If -v is provided, runs an OpenGL visualization of planning"
        exit(0)
    if sys.argv[1] == '-v':
        from pomp.visualizer import runVisualizer
        #visualization mode
        print "Testing visualization with problem",sys.argv[2],"and planner",sys.argv[3]
        runViz(sys.argv[2],sys.argv[3])
    else:
        print
        print "Testing problems",sys.argv[1],"with planners",sys.argv[2:]
        runTests(problems=[sys.argv[1]],planners=sys.argv[2:])
