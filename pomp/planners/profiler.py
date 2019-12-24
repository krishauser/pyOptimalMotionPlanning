from __future__ import print_function,division
from six import iteritems

class CountProfiler:
    """Collects an accumulated item."""
    def __init__(self):
        self.count = 0
    def __str__(self):
        return str(self.count)
    def set(self,value):
        self.count = value
    def add(self,value):
        self.count += value
    def __iadd__(self,value):
        self.add(value)
        return self

class ValueProfiler:
    """Collects a number-valued item, reporting the min, max, mean, and
    variance.  Can also be weighted."""
    def __init__(self):
        self.average = 0
        self.variance = 0
        self.minimum = None
        self.maximum = None
        self.count = 0
    def __str__(self):
        if self.count==0: return 'empty'
        return 'min %f, max %f, average %f, count %f'%(self.minimum,self.maximum,self.average,self.count)
    def reset(self):
        self.average = 0
        self.variance = 0
        self.minimum = None
        self.maximum = None
        self.count = 0        
    def add(self,value,weight=1):
        if self.count==0:
            self.minimum = self.maximum = value
        else:
            if value < self.minimum: self.minimum=value
            elif value > self.maximum: self.maximum=value
        oldEsq =  self.variance + self.average*self.average
        self.average = (self.count*self.average + weight*value)/(self.count+weight)
        newEsq = oldEsq + weight*value*value
        self.variance = newEsq - self.average*self.average
        self.count += weight
    def __iadd__(self,value):
        self.add(value)

class TimingProfiler(ValueProfiler):
    def __init__(self):
        ValueProfiler.__init__(self)
        self.tstart = None
    def begin(self):
        assert self.tstart == None, "Called begin() twice"
        self.tstart = time.time()
    def end(self):
        assert self.tstart != None, "Called end() without begin"
        t = time.time()-self.tstart
        self.tstart = None
        self.add(t)

class Profiler:
    def __init__(self):
        self.items = {}
    def stopwatch(self,item):
        try:
            return self.items[item]
        except KeyError:
            self.items[item] = TimingProfiler()
            return self.items[item]
    def count(self,item):
        try:
            return self.items[item]
        except KeyError:
            self.items[item] = CountProfiler()
            return self.items[item]
    def value(self,item):
        try:
            return self.items[item]
        except KeyError:
            self.items[item] = ValueProfiler()
            return self.items[item]
    def descend(self,item):
        try:
            return self.items[item]
        except KeyError:
            self.items[item] = Profiler()
            return self.items[item]
    def pretty_print(self,indent=0):
        for (k,v) in iteritems(self.items):
            print(' '*indent+str(k),":",end='')
            if isinstance(v,Profiler):
                print()
                v.pretty_print(indent+1)
            else:
                print(str(v))
