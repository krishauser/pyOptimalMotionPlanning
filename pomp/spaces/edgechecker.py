import math

class EdgeChecker:
    def feasible(self,interpolator):
        """ interpolator: a subclass of Interpolator.
        Returns true if all points along the interpolator are feasible.
        """
        raise NotImplementedError()

class EpsilonEdgeChecker(EdgeChecker):
    """An edge checker that uses a fixed resolution for collision checking.
    """
    def __init__(self,space,resolution):
        """Arguments:
            - space: a subclass of ConfigurationSpace
            - resolution: an edge checking resolution
        """
        self.space = space
        self.resolution = resolution
    def feasible(self,interpolator):
        l = interpolator.length()
        k = int(math.ceil(l / self.resolution))
        if not self.space.feasible(interpolator.start()) or not self.space.feasible(interpolator.end()):
            return False
        for i in range(k):
            u = float(i+1)/float(k+2)
            x = interpolator.eval(u)
            if not self.space.feasible(x):
                return False
        return True
