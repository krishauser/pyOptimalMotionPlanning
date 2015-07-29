from OpenGL.GL import *
from geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem

class FlappyControlSpace(ControlSpace):
    def __init__(self,flappy):
        self.flappy = flappy
    def configurationSpace(self):
        return self.flappy.configurationSpace()
    def controlSet(self,x):
        return MultiSet(TimeBiasSet(self.flappy.time_range,self.flappy.controlSet()),self.flappy.controlSet())
        #return MultiSet(BoxSet([0],[self.flappy.time_range]),self.flappy.controlSet())
    def nextState(self,x,u):
        return self.eval(x,u,1.0)
    def eval(self,x,u,amount):
        x_i,y_i,vy_i = x
        t,thrust = u
        tc = t*amount
        #instantaneous version
        #net_acceler = self.flappy.gravity
        #vy_i += thrust*self.flappy.thrust
        #yilun's version
        net_acceler = self.flappy.gravity + thrust*self.flappy.thrust
        return [x_i+self.flappy.v_x*tc, 
                y_i+vy_i*tc+0.5*net_acceler*(tc**2), 
                vy_i+net_acceler*tc]

    def interpolator(self,x,u):
        return LambdaInterpolator(lambda s:self.eval(x,u,s),self.configurationSpace(),10)

class Flappy:
    def __init__(self):
        self.x_range = 1000
        self.y_range = 600
        self.min_altitude = 300
        self.max_velocity = 40

        self.start_state = [50, 250, 0]
        self.goal_state = [950, 200, 0]
        self.goal_radius = 50
        self.time_range = 10
        #u = lambda:round(random.random())

        self.obstacles = []
        self.obstacles = [(175, 450, 50, 100), (175, 0, 50, 100), (175, 150, 50, 200), 
             (375,200, 50, 300),(375, 0, 50, 100), 
             (575, 500, 50, 100), (575, 0, 50, 125), (575, 200, 50, 200), 
             (775, 200, 50, 400)]

        self.v_x = 5
        self.gravity = -1
        self.thrust = 4

    def controlSet(self):
        return FiniteSet([[0],[1]])

    def controlSpace(self):
        return FlappyControlSpace(self)

    def workspace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [0,0]
        wspace.box.bmax = [self.x_range,self.y_range]
        for o in self.obstacles:
            wspace.addObstacle(Box(o[0],o[1],o[0]+o[2],o[1]+o[3]))
        return wspace
    
    def configurationSpace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [0,0]
        wspace.box.bmax = [self.x_range,self.y_range]
        for o in self.obstacles:
            wspace.addObstacle(Box(o[0],o[1],o[0]+o[2],o[1]+o[3]))
        res =  MultiConfigurationSpace(wspace,BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]))
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        r = self.goal_radius
        return BoxSubset(self.configurationSpace(),
                         [self.goal_state[0]-r,self.goal_state[1]-r,-self.max_velocity],
                         [self.goal_state[0]+r,self.goal_state[1]+r,self.max_velocity])


class FlappyObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self,flappy,timestep=0.2):
        self.flappy = flappy
        self.space = flappy.controlSpace()
        self.timestep = timestep
    def incremental(self,x,u):
        e = self.space.interpolator(x,u)
        tmax = u[0]
        t = 0
        c = 0
        while t < tmax:
            t = min(tmax,t+self.timestep)
            xnext = e.eval(t / tmax)
            c += vectorops.distance(x,xnext)
            x = xnext
        return c


def flappyTest():
    p = Flappy()
    objective = FlappyObjectiveFunction(p)
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean = True)


