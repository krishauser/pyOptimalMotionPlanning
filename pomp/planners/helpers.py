import random

infty = float('inf')

def popdefault(mapping,item,default,warning=True):
    """Extracts item from the dict mapping.  If it doesnt exist, returns
    default."""
    try:
        res =  mapping[item]
        del mapping[item]
        return res
    except KeyError:
        if warning == True:
            print "Parameter",item,"not specified, using default",default
        elif warning != None and warning != False:
            print warning
        return default

def arg_max(weights,args = None):
    """Returns the index of the list weights which contains the maximum
    item.  If vals is provided, then this indicates the index"""
    if args == None: args = range(len(weights))
    return max(zip(weights,args))[1]

def cumsum(ls):
    """Returns a list containing the cumulative sums at every element of
    ls.
    
    i.e., cumsum([1,2,3]) = [1,3,6]."""
    
    acc = 0
    r = [0 for v in ls]
    for i,v in enumerate(ls):
        acc += v
	r[i] = acc
    return r

def sample_weighted(weights, vals=None, eps=1.0e-4):
    """Selects a value from vals with probability proportional to the
    corresponding value in weights.

    If vals == None, returns the index that would have been picked
    """
	
    weightSum = sum(weights)
    if weightSum == 0:
        if vals==None:
            return random.randint(0,len(weights)-1)
        return random.choice(vals)
    r = random.uniform(0.0,weightSum)
    if vals==None:
        for i,w in enumerate(weights):
            if r <= w:
                return i
            r -= w
        return len(weights)-1
    else:
        for v,w in zip(vals,weights):
            if r <= w:
                return v
            r -= w
        return vals[-1]
