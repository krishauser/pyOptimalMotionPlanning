from configurationspace import *

class TCSpace(MultiConfigurationSpace):
    """A configuration space that prepends the time variable t to another
    configuration space."""
    def __init__(self,cspace):
        MultiConfigurationSpace.__init__(CartesianConfigurationSpace(1),cspace)
        if hasattr(cspace,'geodesic'):
            self.geodesic = MultiGeodesicSpace(CartesianSpace(1),cspace.geodesic)
        return
