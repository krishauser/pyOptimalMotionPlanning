from OpenGL.GL import *
from .geometric import *
from ..spaces.objectives import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..spaces.biassets import InfiniteBiasSet
from ..planners.problem import PlanningProblem
import numpy as np

class LQRVisualizer:
    def __init__(self,workspace):
        self.base = workspace

    def toScreen(self,q):
        return q[0],q[1]

    def toState(self,x,y):
        return (x,y,0,0)

    def drawObstaclesGL(self):
        self.base.drawObstaclesGL()

    def drawVerticesGL(self,qs):
        self.base.drawVerticesGL(qs)

    def drawRobotGL(self,q):
        glColor3f(0,0,1)
        glPointSize(7.0)
        self.drawVerticesGL([q])

    def drawGoalGL(self,goal):
        #self.base.drawGoalGL(goal)
        pass

    def drawInterpolatorGL(self,interpolator):
        self.base.drawInterpolatorGL(interpolator)


def lqrTest():
    #double integrator test
    n = 2
    m = 1
    cspace = Geometric2DCSpace()
    cspace.box.bmin = [-2,-2]
    cspace.box.bmax = [2,2]
    cspace.addObstacle(Circle(0,0,0.5))
    uspace = InfiniteBiasSet(m,1)
    start = [0.9,1.0]
    goal = InfiniteBiasSet(n,2)
    dt = 0.25
    A = np.zeros((n,n))
    A[0,0] = 1.0
    A[0,1] = dt
    A[1,1] = 1.0
    B = np.zeros((n,m))
    B[0,0] = 0.5*dt**2
    B[1,0] = dt
    Q = np.eye(n)*0.01
    P = np.eye(m)*0.01
    Qterm = np.eye(n)*1.0
    objective = QuadraticObjectiveFunction(Q,np.zeros((n,m)),P,np.zeros(n),np.zeros(m),0,
        Qterm,np.zeros(n))
    controlSpace = LTIControlSpace(cspace,uspace,A,B)
    return PlanningProblem(controlSpace,start,goal,
                           objective=objective,
                           visualizer=LQRVisualizer(cspace),
                           euclidean = True)
