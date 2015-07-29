class Sampler:
    """A base class for a sampling routine."""
    def __init__(self,space):
        self.space = space
    def sample(self):
        return self.space.sample()

class NeighborhoodSampler:
    """A sampler that samples the neighborhood of a point."""
    def __init__(self,space,c,r):
        self.space = space
    def sample(self):
        return self.space.sampleNeighborhood(c,r)

class FeasibleSampler:
    """A sampler that uses rejection sampling to sample a feasible point."""
    def __init__(self,space,maxSamples = 1000):
        self.space = space
        self.maxSamples = 1000
    def sample(self):
        for i in range(self.maxSamples):
            x = self.space.sample()
            if self.space.feasible(x):
                return x
        return self.space.sample()

class SubsetSampler:
    """Given a space and a ConfigurationSubset, samples the subset"""
    def __init__(self,space,subset,maxSamples = 1000):
        self.space = space
        self.subset = subset
        self.maxSamples = maxSamples
    def sample(self):
        res = self.subset.sample()
        if res != None:
            return res
        for i in range(self.maxSamples):
            x = self.space.sample()
            if self.subset.contains(x):
                return x
        return self.space.sample()
