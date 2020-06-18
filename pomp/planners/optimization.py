from __future__ import print_function,division
from builtins import range
from six import iteritems

from ..spaces.objective import ObjectiveFunction
from ..spaces.sets import Set
from ..spaces.controlspace import ControlSpace
from ..spaces.configurationspace import ConfigurationSpace
import math
import numpy as np



class iLQR:
    """An implementation of the iLQR trajectory optimization algorithm.
    
    This performs a conversion of obstacles into smooth barrier functions and
    optimizes the objective function augmented with the barrier value.
    
    Controls are projected onto the feasible control set at each step.
    
    Attributes:
        xref (array of size (T+1,n)): the optimized reference state trajectory
        uref (array of size (T,m)): the optimized reference control trajectory
        gains (pair of arrays): a pair (K,k) of arrays so that for each time step, 
            the optimized control is given by
                
                u(x,t) ~= K[t]*(x - xref[t]) + k[t] + uref[t]
            
            K has dimension (T,m,n) and k has dimension (T,m)
            
        value (triple of T+1-lists): a triple (V,Vx,Vxx) of arrays so that for each
            time step, the quadratic expansion of the value function is given by:
            
                V(x,t) ~= 1/2 dx^T Vxx[t] dx + dx^T Vx[t] + V[t]
            
            with  dx = x-xref[t]. 
        
        costGradients (array of size T,m): the gradients of total cost w.r.t.
            controls.
    """
    def __init__(self,controlSpace,objective,goalSet=None,clearanceToCostFn='square',clearanceToCostWeight=0.1):
        assert isinstance(objective,ObjectiveFunction)
        assert isinstance(goalSet,Set)
        self.controlSpace = controlSpace
        self.rawObjective = objective
        self.cspace = controlSpace.configurationSpace()
        self.goalSet = goalSet
        print("iLQR Objective function",objective,"augmented with",clearanceToCostFn,"barrier")
        if clearanceToCostFn is not None:
            self.objective = BarrierAugmentedObjective(objective,controlSpace,self.goalSet,clearanceToCostFn,clearanceToCostWeight)
        else:
            self.objective = objective
        self.clearanceToCostWeight = clearanceToCostWeight
        self.adaptiveClearanceWeight = False
        self.xref = None
        self.uref = None
        self.gains = None
        self.value = None
        self.costGradients = None

    def run(self,x,u,maxIters,maxInnerIters=10,xtol=1e-7,gtol=1e-7,ftol=1e-7,damping=1e-5):
        if len(u)==0:
            raise ValueError("Cannot optimize with no controls")
        if not isinstance(self.objective,BarrierAugmentedObjective):
            maxInnerIters = 1
        if not hasattr(x[0],'__iter__'):
            #assume its a single state
            x0 = x
            x = [x0]
            for ut in u:
                x.append(self.controlSpace.nextState(x[-1],ut))
        assert len(x) == len(u)+1
        self.xref = np.array([xt for xt in x])
        self.uref = np.array([ut for ut in u])
        T = len(u)
        m = len(u[0])
        n = len(x[0])
        self.gains = (np.zeros((T,m,n)),np.zeros((T,m)))
        self.value = (np.zeros((T+1)),np.zeros((T+1,n)),np.zeros((T+1,n,n)))
        self.costGradients = np.zeros((T,m))
        if isinstance(self.objective,BarrierAugmentedObjective):
            self.objective.setBarrierFromTrajectory(self.xref,self.uref)
            feasible = self.objective.isFeasible()
            if feasible:
                print("iLQR: Starting from a feasible point with clearance for state",self.objective.barrierClearance,"control",self.objective.controlBarrierClearance,"goal",self.objective.goalBarrierClearance)
            else:
                print("iLQR: Starting from an infeasible point with clearance for state",self.objective.barrierClearance,"control",self.objective.controlBarrierClearance,"goal",self.objective.goalBarrierClearance)
        else:
            feasible = True
        if not self.controlSpace.checkDerivatives(x[0],u[0]) or not self.controlSpace.checkDerivatives(x[-2],u[-1]):
            input("Press enter to continue >")
        if not self.objective.checkDerivatives(x[0],u[0]) or not self.objective.checkDerivatives(x[-2],u[-1]):
            input("Press enter to continue >")
        
        #first cost backup
        costTraj = self.value[0]
        costTraj[:] = self.evalCosts(self.xref,self.uref)
        print("INITIAL TRAJECTORY")
        for (a,b) in zip(x,u):
            print("  ",a,b)
        print("  ",x[-1])
        print("COST TRAJECTORY",costTraj)
        print("OBJECTIVE TYPE",self.objective.__class__.__name__)
        J0 = costTraj[0]
        if not math.isfinite(J0):
            raise ValueError("Need to provide a feasible path as input?")
        J0raw = self.rawObjective.cost(self.xref,self.uref)
        print("INITIAL AUGMENTED COST",J0,"TRUE COST",J0raw)
        
        for iter in range(maxIters):
            alpha = 1.0
            for inner in range(maxInnerIters):
                self.backward()
                g = self.costGradients
                gnorm = np.linalg.norm(g)
                if gnorm < gtol:
                    return True,'Convergence to stationary point'
                knorm = np.linalg.norm(self.gains[1])
                print("iLQR: Norm of nominal step size: %.3f, gradient norm %.3f"%(knorm,gnorm))
                if np.dot(g.flatten(),self.gains[1].flatten()) > 0:
                    print("WARNING: LQR step has direction reverse from gradient")
                    self.gains[1][:] = -g
                    knorm = gnorm
                #test gradient descent
                #self.gains[1][:] = -g
                #print("   Gains:",self.gains[1])
                #print("   Gradients",g)
                lineSearchIters = 0
                alpha0 = alpha
                while alpha*knorm > xtol and lineSearchIters < maxIters:
                    lineSearchIters += 1
                    xu = self.forward(alpha)
                    if xu is None:
                        #failure, shrink step size
                        alpha *= 0.5
                        continue
                    x,u = xu
                    Ja = self.evalCosts(x,u,cbranch=J0) 
                    if Ja[0] < J0 and abs(Ja[0]-self.objective.cost(x,u)) > 1e-4:
                        print("Uh... difference in costs?",Ja[0],"vs",self.objective.cost(x,u))
                        input("Press enter to continue >")
                    if Ja[0] < J0:
                        #accept step
                        self.xref = x
                        self.uref = u
                        self.value[0][:] = Ja
                        print("iLQR: Step length %.3g reduced augmented cost to %.3f < %.3f"%(alpha,Ja[0],J0))
                        print("   standard cost changed from %.3f to %.3f"%(J0raw,self.rawObjective.cost(self.xref,self.uref)))
                        #print("   Endpoints",x[0],x[1])
                        #print("   Controls",u)
                        if alpha == alpha0:
                            #succeeded on first step, increase default step size
                            alpha *= 2.5
                            if alpha > 1.0:
                                alpha = 1.0
                        break
                    else:
                        #failure, shrink step size
                        #print("Rejected step to cost",Ja[0])
                        alpha *= 0.5
                        
                self.value[0][:] = Ja
                J0 = Ja[0]
                J0raw = self.rawObjective.cost(self.xref,self.uref)

                if alpha*knorm <= xtol or lineSearchIters == maxIters:
                    print("iLQR: Inner iterations stalled at",lineSearchIters,"LS iters, step size",alpha,", gradient norm",knorm,"< tolerance",xtol)
                    break

            print("iLQR: Outer iteration done, clearance for state",self.objective.barrierClearance,"control",self.objective.controlBarrierClearance,"goal",self.objective.goalBarrierClearance)
            #next outer iteration
            """
            if isinstance(self.objective,BarrierAugmentedObjective):
                self.objective.barrierWeight *= 0.5
                self.objective.controlBarrierWeight *= 0.5
                self.objective.goalBarrierWeight *= 0.5
            """

            if not isinstance(self.objective,BarrierAugmentedObjective) or max(self.objective.barrierWeight,self.objective.controlBarrierWeight,self.objective.goalBarrierWeight) < 1e-4:
                print("   COST",self.rawObjective.cost(self.xref,self.uref))
                return True,'Convergence on x'
            else:
                if isinstance(self.objective,BarrierAugmentedObjective):
                    self.objective.updateBarrierFromTrajectory(self.xref,self.uref)
                    Ja = self.evalCosts(self.xref,self.uref)
                    if self.objective.isFeasible() and abs(Ja[0]-J0) < ftol:
                        return True,'Convergence on f'
            self.value[0][:] = Ja
            J0 = Ja[0]
            J0raw = self.rawObjective.cost(self.xref,self.uref)
            print("AUGMENTED COST",Ja[0],"TRUE COST",J0raw,"FEASIBLE",feasible)
            input()

        print("iLQR: final clearance for state",self.objective.barrierClearance,"control",self.objective.controlBarrierClearance,"goal",self.objective.goalBarrierClearance)
        return False,'Max iters reached'
    
    def evalCosts(self,x,u,cbranch=float('inf')):
        """Returns vector of value function evaluated along trajectory."""
        T = len(u)
        assert T+1 == len(x)
        costs = np.empty(len(x))
        costs[-1] = self.objective.terminal(x[T])
        if costs[-1] > cbranch:
            costs[0] = costs[-1]
            return costs
        for i in range(T)[::-1]:
            xt = x[i]
            ut = u[i]
            c = self.objective.incremental(xt,ut)
            costs[i] = costs[i+1] + c
            if costs[i] > cbranch:
                costs[0] = costs[i]
                return costs
        return costs

    def backward(self,damping=1e-3):
        """Computes the LQR backup centered around self.xref,self.uref.
        
        Will fill out self.gains, self.costGradients, and the 2nd and 3rd
        elements of self.value
        """
        T = len(self.gains[0])
        Vx = self.objective.terminal_gradient(self.xref[T])
        Vxx = self.objective.terminal_hessian(self.xref[T])
        if np.linalg.norm(Vxx-Vxx.T) > 1e-3:
            print("ERROR IN TERMINAL HESSIAN",self.xref[T])
            print(Vxx)
            raise ValueError()
        self.value[1][-1] = Vx
        self.value[2][-1] = Vxx
        print("iLQR BACKWARDS PASS")
        #print("   Terminal cost",self.objective.terminal(self.xref[T]))
        #print("   Terminal grad",Vx)
        #print("   Terminal Hessian",Vxx)
        for i in range(T)[::-1]:
            #print("timestep",i)
            xt,ut = self.xref[i],self.uref[i]
            fx,fu = self.controlSpace.nextState_jacobian(xt,ut)
            cx,cu = self.objective.incremental_gradient(xt,ut)
            cxx,cxu,cuu = self.objective.incremental_hessian(xt,ut)
            #print("  Next state jacobian x",fx)
            #print("  Next state jacobian u",fu)
            Qxx = fx.T.dot(Vxx.dot(fx))+cxx
            Quu = fu.T.dot(Vxx.dot(fu))+cuu
            Qxu = fx.T.dot(Vxx.dot(fu))+cxu
            Vxc = Vx
            Qx = cx + fx.T.dot(Vxc)
            Qu = cu + fu.T.dot(Vxc)
            if damping > 0:
                Quu = (Quu + Quu.T)*0.5
                Quu_evals, Quu_evecs = np.linalg.eig(Quu)
                Quu_evals[Quu_evals < 0] = 0.0
                Quu_evals += damping
                QuuInv = np.dot(Quu_evecs,np.dot(np.diag(1.0/Quu_evals),Quu_evecs.T))
            else:
                QuuInv = np.linalg.pinv(Quu)
            K = -QuuInv.dot(Qxu.T)
            k = -QuuInv.dot(Qu)
            temp = Qxu.dot(K)
            Vxx = Qxx + temp + temp.T + K.T.dot(Quu.dot(K))
            Vx = Qx + Qxu.dot(k) + K.T.dot(Qu+Quu.dot(k))
            #print("   Vf grad",Vx)
            #print("   Vf Hessian",Vxx)
            self.gains[0][i] = K
            self.gains[1][i] = k
            self.value[1][i] = Vx
            self.value[2][i] = Vxx
            self.costGradients[i] = Qu

    def forward(self,alpha=1.0):
        """Computes the iLQR forward pass, assuming the gain matrices have been computed"""
        x = np.empty(self.xref.shape)
        u = np.empty(self.uref.shape)
        x[0] = self.xref[0]
        u[0] = self.uref[0]
        K,k = self.gains
        for i in range(self.uref.shape[0]):
            if i == 0:
                du = k[0]
            else:
                du = k[i] + K[i].dot(x[i]-self.xref[i])
            u[i] = self.uref[i] + alpha*du
            """
            if not self.controlSpace.controlSet(x[i]).contains(u[i]):
                try:
                    ui = self.controlSpace.controlSet(x[i]).project(list(u[i]))
                    if ui is None:
                        print("Projection of control failed?")
                        return None
                    u[i] = ui
                except NotImplementedError:
                    #projection may not be implemented... TODO: address control constraints some other way
                    pass
            """
            x[i+1] = self.controlSpace.nextState(x[i],u[i])
        return (x,u)
        
        
class BarrierAugmentedObjective(ObjectiveFunction):
    def __init__(self,base,controlSpace,goalSet,barrierType,barrierWeight):
        """Barrier types can be 'log', 'inv', 'square'.
        
        Barrier function depends on distance(s) to constraints.
        - 'log': -log(d) if d > 0, inf otherwise
        - 'inv': 1/d if d > 0, inf otherwise
        - 'square': 0 if d > 0, d^2 otherwise (soft constraint)
        """
        self.base = base
        if isinstance(controlSpace,ControlSpace):
            self.controlSpace = controlSpace
            self.cspace = controlSpace.configurationSpace()
        else:
            assert isinstance(controlSpace,ConfigurationSpace)
            self.controlSpace = None
            self.cspace = controlSpace
        self.goalSet = goalSet
        self.barrierType = barrierType
        self.barrierWeight = barrierWeight
        self.barrierClearance = 0.0
        self.barrierShift = 0.0
        self.controlBarrierWeight = barrierWeight
        self.controlBarrierClearance = 0.0
        self.controlBarrierShift = 0.0
        self.goalBarrierWeight = barrierWeight
        self.goalBarrierClearance = 0.0
        self.goalBarrierShift = 0.0
    
    def __str__(self):
        return "Barrier-augmented "+str(self.base)+" barrier "+self.barrierType
    
    def isHard(self):
        return self.barrierType in ['inv','log']
        
    def setBarrierFromTrajectory(self,x,u,scale=1.5,mindist=1e-5,firstTime=True):
        """Evaluates the trajectory clearance and sets the barrier
        offset from a trajectory, ensuring that
          - If the barrier is hard, x,u is feasible under the shifted barrier
          - If x,u is invalid, then the max (unweighted) barrier cost evaluated
            at x,u is equal to `scale`
        
        `mindist` is used so that if the initial point is not strictly
        feasible (or is very close to the boundary) then a positive slack is given
        to the constraint.
        """
        dmin = None
        dumin = float('inf')
        for xi,ui in zip(x,u):
            d = self.cspace.clearance(xi)
            if dmin is None:
                dmin = np.asarray(d)
            else:
                dmin = np.minimum(dmin,d)
            if self.controlSpace is not None:
                U = self.controlSpace.controlSet(xi)
                dumin = min(dumin,-U.signedDistance(ui))
        self.barrierClearance = dmin
        if self.controlSpace is None:
            self.controlBarrierClearance = 0
        else:
            self.controlBarrierClearance = dumin
        if self.goalSet is not None:
            self.goalBarrierClearance = -self.goalSet.signedDistance(x[-1])
        else:
            self.goalBarrierClearance = 0.0
        if self.isHard():
            if not firstTime:
                oldBarrierShift = self.barrierShift
                oldControlBarrierShift = self.controlBarrierShift
                oldGoalBarrierShift = self.goalBarrierShift
            self.barrierShift = np.minimum(self.barrierClearance,0.0)
            self.controlBarrierShift = min(self.controlBarrierClearance,0.0)
            self.goalBarrierShift = min(self.goalBarrierClearance,0.0)
            if self.barrierType == 'inv':
                #scale = 1/(barrierClearance - barrierShift) => barrierClearance - 1/scale = barrierShift
                self.barrierShift[self.barrierClearance < mindist] -= 1.0/scale
                if self.controlBarrierClearance < mindist: self.controlBarrierShift -= 1.0/scale
                if self.goalBarrierClearance < mindist: self.goalBarrierShift -= 1.0/scale
            elif self.barrierType == 'log':
                #scale = -log(barrierClearance - barrierShift) => barrierShift = barrierClearance - exp(-scale)
                print(self.barrierClearance < mindist,self.barrierClearance,mindist)
                self.barrierShift[self.barrierClearance < mindist] -= math.exp(-scale)
                if self.controlBarrierClearance < mindist: self.controlBarrierShift -= math.exp(-scale)
                if self.goalBarrierClearance < mindist: self.goalBarrierShift -= math.exp(-scale)
            else:
                raise ValueError("Invalid barrier string, only log, inv, and square are supported")
            if not firstTime:
                self.barrierShift = np.maximum(self.barrierShift,oldBarrierShift)
                self.controlBarrierShift = max(self.controlBarrierShift,oldControlBarrierShift)
                self.goalBarrierShift = max(self.goalBarrierShift,oldGoalBarrierShift)
            print("Barrier clearances",self.barrierClearance,self.controlBarrierClearance,self.goalBarrierClearance)
            print("Barrier shifts: state",self.barrierShift,"control",self.controlBarrierShift,"goal",self.goalBarrierShift)
            print("   => Cost",self.cost(x,u))
            input()
        else:
            self.barrierShift = 0.0
            self.controlBarrierShift = 0.0
            self.goalBarrierShift = 0.0
    
    def isFeasible(self):
        return all(v >= 0 for v in self.barrierClearance) and self.controlBarrierClearance >= 0 and self.goalBarrierClearance >= 0
        
    def updateBarrierFromTrajectory(self,x,u):
        oldbc = self.barrierClearance
        oldcbc = self.controlBarrierClearance
        oldgbc = self.goalBarrierClearance
        self.setBarrierFromTrajectory(x,u,firstTime=False)
        print("iLQR: clearance on state",self.barrierClearance,"control",self.controlBarrierClearance,"goal",self.goalBarrierClearance)
        cold = min(oldbc)
        c = min(self.barrierClearance)
        if c >= 0:
            if cold < 0:
                print("iLQR: Switched from infeasible to feasible on state constraint, clearance %.3g -> %.3g"%(cold,c))
            else:
                self.barrierWeight *= 0.5
                print("iLQR: Stayed feasible on state constraint, sharpening constraint to %0.3g"%(self.barrierWeight,))
        else:
            if cold < 0:
                self.barrierWeight *= 2.5
                print("iLQR: Stayed infeasible on state constraint, diffusing constraint to %0.3g"%(self.barrierWeight,))
            else:
                print("iLQR: Switched from feasible to infeasible on state constraint, clearance %.3g -> %.3g"%(cold,c))
        cold = oldcbc
        c = self.controlBarrierClearance
        if c >= 0:
            if cold < 0:
                print("iLQR: Switched from infeasible to feasible on control constraint, clearance %.3g -> %.3g"%(cold,c))
            else:
                self.controlBarrierWeight  *= 0.5
                print("iLQR: Stayed feasible on control constraint, sharpening constraint to %0.3g"%(self.controlBarrierWeight,))
        else:
            if cold < 0:
                self.controlBarrierWeight *= 2.5
                print("iLQR: Stayed infeasible on control constraint, diffusing constraint to %0.3g"%(self.controlBarrierWeight,))
            else:
                print("iLQR: Switched from feasible to infeasible on control constraint, clearance %.3g -> %.3g"%(cold,c))
        cold = oldgbc
        c = self.goalBarrierClearance
        if c >= 0:
            if cold < 0:
                print("iLQR: Switched from infeasible to feasible on goal constraint, clearance %.3g -> %.3g"%(cold,c))
            else:
                self.goalBarrierWeight  *= 0.5
                print("iLQR: Stayed feasible on goal constraint, sharpening constraint to %0.3g"%(self.goalBarrierWeight,))
        else:
            if cold < 0:
                self.goalBarrierWeight *= 2.5
                print("iLQR: Stayed infeasible on goal constraint, diffusing constraint to %0.3g"%(self.goalBarrierWeight,))
            else:
                print("iLQR: Switched from feasible to infeasible on goal constraint, clearance %.3g -> %.3g"%(cold,c))
    
    def barrierFn(self,c):
        if self.barrierType == 'inv':
            if c <= 0: return float('inf')
            return 1.0/c
        elif self.barrierType == 'log':
            if c <= 0: return float('inf')
            if math.isinf(c): return 0
            return -math.log(c)
        elif self.barrierType == 'square':
            if c < 0: return c**2
            return 0
        else:
            raise ValueError("Invalid barrier function")
            
    def barrierDeriv(self,c):
        if self.barrierType == 'inv':
            if c <= 0: dc = 0.0
            else: dc= -1.0/c**2
        elif self.barrierType == 'log':
            if c <= 0: dc = 0.0
            elif math.isinf(c): dc = 0.0
            else: dc = -1.0/c
        elif self.barrierType == 'square':
            if c < 0: dc = 2*c
            else: dc = 0
        return dc
        
    def barrierDeriv2(self,c):
        if self.barrierType == 'inv':
            if c <= 0: dc = 0.0
            else: dc= 2.0/c**3
        elif self.barrierType == 'log':
            if c <= 0: dc = 0.0
            elif math.isinf(c): dc = 0.0
            else: dc = 1.0/c**2
        elif self.barrierType == 'square':
            if c < 0: dc = 2
            else: dc = 0
        return dc
        
    def barrier(self,x):
        c = self.cspace.clearance(x) - self.barrierShift
        if hasattr(c,'__iter__'):
            return self.barrierWeight*sum(self.barrierFn(v) for v in c)
        else:
            return self.barrierWeight*self.barrierFn(c)
    
    def barrier_gradient(self,x):
        c = self.cspace.clearance(x) - self.barrierShift
        g = self.cspace.clearance_gradient(x)
        if hasattr(c,'__iter__'):
            return self.barrierWeight*sum(self.barrierDeriv(v)*gi for v,gi in zip(c,g))
        else:
            return self.barrierWeight*self.barrierDeriv(c)*g
    
    def barrier_hessian(self,x):
        c = self.cspace.clearance(x) - self.barrierShift
        g = self.cspace.clearance_gradient(x)
        if hasattr(c,'__iter__'):
            return self.barrierWeight*sum(self.barrierDeriv2(v)*np.outer(gi,gi) for v,gi in zip(c,g))
        else:
            return self.barrierWeight*self.barrierDeriv2(c)*np.outer(g,g)

    def controlBarrier(self,x,u):
        if self.controlSpace is None: return 0.0
        U = self.controlSpace.controlSet(x)
        c = -U.signedDistance(u) - self.controlBarrierShift
        if U.signedDistance(u) <= 0:
            assert U.contains(u),"Control set %s signed distance %f but doesn't contain %s"%(str(U),U.signedDistance(u),str(u))
        else:
            assert not U.contains(u),"Control set %s signed distance %f but contains %s"%(str(U),U.signedDistance(u),str(u))
        return self.controlBarrierWeight*self.barrierFn(c)
    
    def controlBarrier_gradient(self,x,u):
        if self.controlSpace is None: return None
        U = self.controlSpace.controlSet(x)
        c = -U.signedDistance(u) - self.controlBarrierShift
        g = -U.signedDistance_gradient(u)
        return self.controlBarrierWeight*self.barrierDeriv(c)*g
    
    def controlBarrier_hessian(self,x,u):
        if self.controlSpace is None: return None
        U = self.controlSpace.controlSet(x)
        c = -U.signedDistance(u) - self.controlBarrierShift
        g = -U.signedDistance_gradient(u)
        return self.controlBarrierWeight*self.barrierDeriv2(c)*np.outer(g,g)
    
    def goalBarrier(self,x):
        c = -self.goalSet.signedDistance(x) - self.goalBarrierShift
        return self.goalBarrierWeight*self.barrierFn(c)
    
    def goalBarrier_gradient(self,x):
        c = -self.goalSet.signedDistance(x) - self.goalBarrierShift
        g = -self.goalSet.signedDistance_gradient(x)
        return self.goalBarrierWeight*self.barrierDeriv(c)*g
    
    def goalBarrier_hessian(self,x):
        c = -self.goalSet.signedDistance(x) - self.goalBarrierShift
        g = -self.goalSet.signedDistance_gradient(x)
        return self.goalBarrierWeight*self.barrierDeriv2(c)*np.outer(g,g)
        
    def incremental(self,x,u=None):
        res = self.base.incremental(x,u)+self.barrier(x)
        if u is not None and self.controlSpace is not None:
            res += self.controlBarrier(x,u)
        return res
    
    def terminal(self,x):
        if self.goalSet is not None:
            return self.base.terminal(x)+self.goalBarrier(x)+self.barrier(x)
        return self.base.terminal(x)+self.barrier(x)
    
    """
    def incremental_gradient(self,x,u):
        return self.incremental_gradient_diff(x,u)
    def incremental_hessian(self,x,u):
        return self.incremental_hessian_diff(x,u)
    def terminal_gradient(self,x):
        return self.terminal_gradient_diff(x)
    def terminal_hessian(self,x):
        return self.terminal_hessian_diff(x)
    """
    def incremental_gradient(self,x,u):
        bx,bu = self.base.incremental_gradient(x,u)
        if u is not None and self.controlSpace is not None:
            bu += self.controlBarrier_gradient(x,u)
        return bx+self.barrier_gradient(x),bu
    
    def incremental_hessian(self,x,u):
        Hx,Hxu,Hu = self.base.incremental_hessian(x,u)
        if u is not None and self.controlSpace is not None:
            Hu += self.controlBarrier_hessian(x,u)
        return Hx+self.barrier_hessian(x),Hxu,Hu
    
    def terminal_gradient(self,x):
        if self.goalSet is not None:
            return self.base.terminal_gradient(x)+self.goalBarrier_gradient(x)+self.barrier_gradient(x)
        return self.base.terminal_gradient(x)+self.barrier_gradient(x)
    
    def terminal_hessian(self,x):
        if self.goalSet is not None:
            return self.base.terminal_hessian(x)+self.goalBarrier_hessian(x)+self.barrier_hessian(x)
        return self.base.terminal_hessian(x)+self.barrier_hessian(x)
