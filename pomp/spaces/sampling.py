import random
from ..klampt import vectorops

def sample_hypersphere(d,c,r):
    """Samples a d-dimensional sphere uniformly, centered at c and with
    radius r"""
    assert(d == len(c))
    d = [random.gauss(0,1) for ci in c]
    d = vectorops.unit(d)
    return vectorops.madd(c,d,r)


def sample_hyperball(d,c,r):
    """Samples a d-dimensional ball uniformly, centered at c and with
    radius r"""
    assert(d == len(c))
    rad = pow(random.random(),1.0/d)
    return sample_hypersphere(d,c,rad)
