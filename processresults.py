#!/usr/bin/env python
import sys
import os
import glob
import csv
import numpy as np
import bisect
from collections import defaultdict

if len(sys.argv) < 2:
    print "Usage: processresults.py folder [outfile]"
    exit(0)

trialvarname = 'trial'
timevarname = 'plan time'
ignorevars = [trialvarname,timevarname,'plan iters','numComponents','numIters','gridMin','gridMax']
#only output means for the following
meanvars = ['configCheckTime','knnTime','connectTime','lazyTime','lazyPathCheckTime','shortestPathsTime','numEdgeChecks','numEdgesPrechecked','numMilestones']

class Trajectory:
    def __init__(self):
        self.times = []
        self.values = []
    def append(self,t,v):
        self.times.append(t)
        self.values.append(v)
    def eval(self,t):
        """Evaluates the trajectory"""
        (i,u) = self.getSegment(t)
        if i<0: return self.values[0]
        elif i>=len(self.values): return self.values[-1]
        #piecewise constant interpolation
        #return self.values[i]
        #linear constant interpolation
        return self.values[i] + u*(self.values[i+1]-self.values[i])
    def getSegment(self,t):
        """Returns the index and interpolation parameter for the
        segment at time t."""
        if len(self.times)==0:
            raise ValueError("Empty trajectory")
        if len(self.times)==1:
            return (-1,0)
        if t > self.times[-1]:
            return (len(self.times),0)
        if t < self.times[0]:
            return (0,0)
        i = bisect.bisect_right(self.times,t)
        p=i-1
        u=(t-self.times[p])/(self.times[i]-self.times[p])
        if i==0:
            return (-1,0)
        assert u >= 0 and u <= 1
        return (p,u)

def parse_data(csvfn):
    with open(csvfn,'r') as f:
        reader = csv.DictReader(f)
        itemtraces = defaultdict(lambda:defaultdict(Trajectory))
        maxtime = 0
        for row in reader:
            trial = int(row[trialvarname])
            time = float(row[timevarname])
            maxtime = max(time,maxtime)
            for (k,v) in row.iteritems():
                if k in ignorevars: continue
                trace = itemtraces[k][trial]
                trace.append(time,float(v))
        res = defaultdict(list)
        dt = 0.1
        t = 0
        while t < maxtime:
            res['time'].append(t)
            for k,trials in itemtraces.iteritems():
                data = [trace.eval(t) for trace in trials.values()]
                cleandata = [v for v in data if np.isfinite(v)]
                if k == 'best cost':
                    successRate = np.mean([1 if np.isfinite(v) else 0 for v in data])
                    res['success fraction'].append(successRate)
                if k in meanvars:
                    if len(cleandata) > 0:
                        res[k].append(np.mean(cleandata))
                    else:
                        res[k].append(None)
                else:
                    if len(cleandata) > 0:
                        res[k+' mean'].append(np.mean(cleandata))
                        res[k+' std'].append(np.std(cleandata))
                        res[k+' min'].append(min(cleandata))
                        if len(data) != len(cleandata):
                            res[k+' max'].append(float('inf'))
                        else:
                            res[k+' max'].append(max(cleandata))
                    else:
                        res[k+' mean'].append(None)
                        res[k+' std'].append(None)
                        res[k+' min'].append(None)
                        res[k+' max'].append(None)
            t += dt
        return res


def process(folder,outfile=None):
    csvfiles = glob.glob(os.path.join(folder,"*.csv"))
    data = dict()
    print "Files in %s:"%(folder)
    for fn in csvfiles:
        name = os.path.splitext(os.path.basename(fn))[0]
        if name == 'summary': continue
        print "  "+name
        fdata = parse_data(fn)
        data[name] = fdata
    fn = (outfile if outfile is not None else os.path.join(folder,"summary.csv"))
    print "Saving summary statistics to",fn
    with open(fn,'w') as f:
        headers = []
        for (name,fdata) in data.iteritems():
            for (k,v) in fdata.iteritems():
                headers.append(name+' '+k)
        writer=csv.DictWriter(f,sorted(headers))
        writer.writeheader()
        i = 0
        while True:
            item = dict()
            for (name,fdata) in data.iteritems():
                for (k,v) in fdata.iteritems():
                    if i < len(v):
                        item[name+' '+k] = v[i]
            if len(item)==0: break
            writer.writerow(item)
            i+=1
    print "Done."

if sys.argv[1] == 'all':
    files = glob.glob(os.path.join('data','*'))
    for f in files:
        process(f)
elif len(sys.argv) >= 3:
    process(sys.argv[1],sys.argv[2])
else:
    process(sys.argv[1])
