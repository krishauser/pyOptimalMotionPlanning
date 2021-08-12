import math
from ..klampt import vectorops

def euclideanMetric(a,b):
    return vectorops.distance(a,b)

def L2Metric(a,b):
    return euclideanMetric(a,b)

def L1Metric(a,b):
    return sum(abs(ai-bi) for (ai,bi) in zip(a,b))

def LinfMetric(a,b):
    return max(abs(ai-bi) for (ai,bi) in zip(a,b))

class WeightedEuclideanMetric:
    def __init__(self,w):
        self.w = w
    def __call__(self,a,b):
        if hasattr(self.w,'__iter__'):
            return math.sqrt(sum(w*(ai-bi)**2 for (w,ai,bi) in zip(self.w,a,b)))
        else:
            return self.w*euclideanMetric(a,b)
                
class LpMetric:
    def __init__(self,p):
        self.p = p
    def __call__(self,a,b):
        return pow(sum(pow(abs(ai-bi),p) for (w,ai,bi) in zip(self.w,a,b)),1.0/p)
