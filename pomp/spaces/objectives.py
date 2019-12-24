from __future__ import print_function,division
from builtins import range
from six import iteritems

from ..klampt import vectorops
from . import differences
from .objective import ObjectiveFunction
import numpy as np


class PathLengthObjectiveFunction(ObjectiveFunction):
    """Meant for a kinematic planning problem: measures path length.
    Assumes the convention of a ControlSpaceAdaptor class, where the control u
    is the next state.
    
    Path cost is sum_{i=0}^{n-1} ||x[i+1]-x[i]||.
    
    Incremental cost is ||x-u||.
    
    For numerical optimization solvers, use the EnergyObjectiveFunction
    which obtains similar results but is much more numerically-friendly.
    """
    def __str__(self):
        return "sum ||dx||"
    def incremental(self,x,u):
        return vectorops.distance(x,u)
    def incremental_gradient(self,x,u):
        #Function is ||x-u|| = sqrt(v^T v) with v=x-u
        #derivative w.r.t. x is v/||v|| 
        d = np.array(x)-u
        g = d/np.linalg.norm(d)
        return g,-g
    def incremental_hessian(self,x,u):
        #derivative w.r.t. x is v/||v|| with v=x-u
        #hessian w.r.t. xx is (dv/dx ||v|| - v d||v||/dx )/ ||v||^2 = 
        #   I/||v|| - vv^T/||v||^3
        d = np.array(x)-u
        dnorm = np.linalg.norm(d)
        g = d/dnorm
        H = np.eye(len(d))/dnorm - np.outer(d,d)/dnorm**3
        return H,-H,H


class EnergyObjectiveFunction(ObjectiveFunction):
    """Meant for a kinematic planning problem: measures integral of squared
    path length.  Assumes the convention of a ControlSpaceAdaptor class, where
    the control u is the next state.
    
    Path cost is sum_{i=0}^{n-1} ||x[i+1]-x[i]||^2.
    
    Incremental cost is ||x-u||^2.
    
    For numerical optimization solvers, use the EnergyObjectiveFunction
    which obtains similar results but is much more numerically-friendly.
    """
    def __str__(self):
        return "sum ||dx||^2"
    def incremental(self,x,u):
        return vectorops.distance(x,u)**2
    def incremental_gradient(self,x,u):
        d = np.array(x)-u
        g = 2*d
        return g,-g
    def incremental_hessian(self,x,u):
        H = 2*np.eye(len(x))
        return H,-H,H



class StepCountObjectiveFunction(ObjectiveFunction):
    """Counts the number of steps until the goal is reached."""
    def __str__(self):
        return "Nsteps"
    def incremental(self,x,u):
        return 1


class TimeObjectiveFunction(ObjectiveFunction):
    """Integrates time step dt.  Meant for a KinodynamicSpace class"""
    def __str__(self):
        return "T"
    def incremental(self,x,u):
        return u[0]
    def incremental_gradient(self,x,u):
        gu = np.zeros(len(u))
        gu[0] = 1
        return np.zeros(len(x)),gu


class QuadraticObjectiveFunction(ObjectiveFunction):
    """A quadratic objective function. 
    
    Incremental cost has the form:
    
      1/2 x^T P x + p^T x + 1/2 u^T Q u + q^T u + x^T R u + s
    
    Terminal cost has the form:
    
      1/2 x^T A x + b^T x + c
      
    The terms in the above equations are given by arguments provided
    to __init__
        - inc_xx: P
        - inc_xu: R
        - inc_uu: Q
        - inc_x: p
        - inc_u: q
        - inc_0: s
        - term_xx: A
        - term_x: b
        - term_0: c
    """
    def __init__(self,inc_xx,inc_xu,inc_uu,inc_x,inc_u,inc_0,
        term_xx,term_x,term_0=0):
        self.inc_xx = inc_xx
        self.inc_xu = inc_xu
        self.inc_uu = inc_uu
        self.inc_x = inc_x
        self.inc_u = inc_u
        self.inc_0 = inc_0
        self.term_xx = term_xx
        self.term_x = term_x
        self.term_0 = term_0
    
    def incremental(self,x,u=None):
        return 0.5*(np.dot(x,np.dot(self.inc_xx,x))+np.dot(u,np.dot(self.inc_uu,u))) + np.dot(x,np.dot(self.inc_xu,u)) + np.dot(self.inc_x,x) + np.dot(self.inc_u,u) + self.inc_0
    def terminal(self,x):
        return 0.5*np.dot(x,np.dot(self.term_xx,x)) + np.dot(self.term_x,x) + self.term_0
    def incremental_gradient(self,x,u):
        gx = np.dot(self.inc_xx,x)+0.5*np.dot(self.inc_xu,u)+self.inc_x
        gu = np.dot(self.inc_uu,u)+0.5*np.dot(self.inc_xu.T,x)+self.inc_u
        return gx,gu
    def incremental_hessian(self,x,u):
        return self.inc_xx,self.inc_xu,self.inc_uu
    def terminal_gradient(self,x):
        return np.dot(self.term_xx,x) + self.term_x
    def terminal_hessian(self,x):
        return self.term_xx

class GoalDistanceObjectiveFunction(ObjectiveFunction):
    """Returns the distance between the terminal state and a goal state.
    
    Can provide a weighting vector or matrix, if desired.
    """
    def __init__(self,xgoal,weight=None):
        self.xgoal = xgoal
        self.weight = weight
        if weight is not None:
            self.weight = np.asarray(weight)
    def __str__(self):
        return "||xT-"+str(self.xgoal)+"||"
    def terminal(self,x):
        d = np.asarray(x)-self.xgoal
        if self.weight is None:
            return np.linalg.norm(d)
        elif len(self.weight.shape) == 2:
            return math.sqrt(np.dot(d,self.weight.dot(d)))
        else:
            return math.sqrt(np.dot(d,np.multiply(self.weight,d)))
    def terminal_gradient(self,x):
        d = np.asarray(x)-self.xgoal
        if self.weight is None:
            return d/np.linalg.norm(d)
        elif len(self.weight.shape) == 2:
            wd = self.weight.dot(d)
            return wd/math.sqrt(np.dot(d,wd))
        else:
            wd = np.multiply(self.weight,d)
            return wd/math.sqrt(np.dot(d,wd))
    def terminal_hessian(self,x):
        d = np.asarray(x)-self.xgoal
        if self.weight is None:
            dnorm = np.linalg.norm(d)
            return np.eye(len(d))/dnorm - 2*np.outer(d,d)/dnorm**3
        elif len(self.weight.shape) == 2:
            wd = self.weight.dot(d)
            dnorm = math.sqrt(np.dot(d,wd))
            return np.diag(self.weight)/dnorm - 2*np.outer(d,wd)/dnorm**3
        else:
            wd = np.multiply(self.weight,d)
            math.sqrt(np.dot(d,wd))
            return self.weight/dnorm - 2*np.outer(d,wd)/dnorm**3


class SetDistanceObjectiveFunction(ObjectiveFunction):
    """Returns the distance between the terminal state and a goal set.
    """
    def __init__(self,goalSet):
        self.goalSet = goalSet
    def __str__(self):
        return "d(xT,"+str(self.goalSet)+")"
    def terminal(self,x):
        return max(self.goalSet.signedDistance(x),0)
    def terminal_gradient(self,x):
        d = self.goalSet.signedDistance(x)
        if d < 0: return np.zeros(len(x))
        return self.goalSet.signedDistance_gradient(x)
    def terminal_hessian(self,x):
        from . import difference
        d = self.goalSet.signedDistance(x)
        if d < 0: return np.zeros(len(x))
        return difference.jacobian_forward_difference(self.goalSet.signedDistance_gradient,x,1e-4)


class TrackingObjectiveFunction(ObjectiveFunction):
    """Integrates tracking error for a timed kinodynamic space. 
    Assumes x[0] is the time variable, x[1:] is the state variable,
    and u[0] is the time step dt.
    
    The tracked trajectory is given by the Trajectory traj.  Q
    is a quadratic penalty matrix.
    """
    def __init__(self,traj,Q,Qterm=None):
        self.traj  = traj
        self.Q = Q
        self.Qterm = Qterm
        if Qterm is None:
            self.Qterm = np.zeros(Q.shape)
    def incremental(self,x,u):
        t = x[0]
        y = x[1:]
        dt = u[0]
        z = self.traj.eval(t)
        d = np.asarray(y)-z
        return dt*0.5*np.dot(d,np.dot(self.Q,d))
    def terminal(self,x):
        t = x[0]
        y = x[1:]
        z = self.traj.eval(t)
        d = np.asarray(y)-z
        return 0.5*np.dot(d,np.dot(self.Qterm,d)) 
    def incremental_gradient(self,x,u):
        t = x[0]
        y = x[1:]
        dt = u[0]
        z = self.traj.eval(t)
        d = np.asarray(y)-z
        dz = self.traj.deriv(t)
        gx = dt*np.hstack(([-np.dot(d,np.dot(self.Q,dz))],np.dot(self.Q,d)))
        gu = np.zeros(len(u))
        gu[0] = 0.5*np.dot(d,np.dot(self.Q,d))
        return gx,gu
    def terminal_gradient(self,x):
        t = x[0]
        y = x[1:]
        z = self.traj.eval(t)
        d = np.asarray(y)-z
        dz = self.traj.deriv(t)
        return np.hstack(([-np.dot(d,np.dot(self.Q,dz))],np.dot(self.Qterm,d)))
    def incremental_hessian(self,x,u):
        t = x[0]
        y = x[1:]
        dt = u[0]
        z = self.traj.eval(t)
        d = np.asarray(y)-z
        dz = self.traj.deriv(t)
        Hxx = dt*np.block([[[-np.dot(dz,np.dot(self.Q,dz))],np.dot(self.Q,dz)],[np.dot(self.Q,dz).T,self.Q]])
        Hxu = np.zeros((len(x),len(u)))
        Hxu[:,0] = np.hstack(([-np.dot(d,np.dot(self.Q,dz))],np.dot(self.Q,d)))
        Huu = np.zeros((len(u),len(u)))
        return Hxx,Hxu,Huu