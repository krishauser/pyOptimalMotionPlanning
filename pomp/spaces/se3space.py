from .so2space import *
from .configurationspace import *

class SE2Space(MultiGeodesicSpace,MultiConfigurationSpace):
    def __init__(self):
        MultiGeodesicSpace.__init__(self,CartesianConfigurationSpace(2),SO2Space())
        MultiConfigurationSpace.__init__(self,CartesianConfigurationSpace(2),SO2Space())
        self.componentWeights = [1,1]
    def setTranslationDistanceWeight(self,value):
        self.componentWeights[0] = value
    def setRotationDistanceWeight(self,value):
        self.componentWeights[1] = value
