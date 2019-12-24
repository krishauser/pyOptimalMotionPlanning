from __future__ import print_function,division
from six import iteritems,iterkeys

import random

class RandomDict(object):
    def __init__(self): # O(1)
        self.dictionary = {}
        self.indexdict = {}
        self.next_index = 0
        self.removed_indices = None
        self.len = 0

    def __len__(self): # might as well include this
        return self.len

    def __contains__(self, key):
        return key in self.dictionary

    def __getitem__(self, key): # O(1)
        return self.dictionary[key][1]

    def __setitem__(self, key, value): # O(1)
        if key in self.dictionary: # O(1)
            self.dictionary[key][1] = value # O(1)
            return
        if self.removed_indices is None:
            index = self.next_index
            self.next_index += 1
        else:
            index = self.removed_indices[0]
            self.removed_indices = self.removed_indices[1]
        self.dictionary[key] = [index, value] # O(1)
        self.indexdict[index] = key # O(1)
        self.len += 1

    def get(self, key, defaultValue): # O(1)
        v = self.dictionary.get(key,None)
        if v == None: return defaultValue
        return v[1]

    def setdefault(self,key,defaultValue):
        if key in self.dictionary: # O(1)
            return self.dictionary[key][1] # O(1)
        if self.removed_indices is None:
            index = self.next_index
            self.next_index += 1
        else:
            index = self.removed_indices[0]
            self.removed_indices = self.removed_indices[1]
        self.dictionary[key] = [index, defaultValue] # O(1)
        self.indexdict[index] = key # O(1)
        self.len += 1
        return self.dictionary[key][1]

    def __delitem__(self, key): # O(1)
        index = self.dictionary[key][0] # O(1)
        del self.dictionary[key] # O(1)
        del self.indexdict[index] # O(1)
        self.removed_indices = (index, self.removed_indices)
        self.len -= 1

    def random_key(self,weight=None):
        """Randomly select a key in the dictionary.  Unweighted
        version is O(log(next_index/len)).  Weighted version selects
        items proportionally to weight(key,value), and runs in time
        O(len)
        """
        if self.len == 0: # which is usually close to O(1)
            raise KeyError
        if weight==None:
            while True:
                r = random.randrange(0, self.next_index)
                if r in self.indexdict:
                    return self.indexdict[r]
        else:
            weights = {}
            sumweight = 0.0
            for k,v in iteritems(self.dictionary):
                w = weight(k,v)
                weights[k] = w
                sumweight += weights[k]
            u = random.random()*sumweight
            for k,v in iteritems(self.dictionary):
                u -= weights[k]
                if u <= 0: return k
            print("Numerical error in random_key")
            return iterkeys(self.dictionary).next()
