#!/usr/bin/env python
from __future__ import print_function,division
from six import iteritems
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

if len(sys.argv) < 3:
    print("Usage: viewsummary.py csvfile item")
    exit(0)

successFraction = 0.5

labelmap = {"lazy_rrgstar":"Lazy-RRG*",
            "lazy_birrgstar":"Lazy-BiRRG*",
            "prmstar":"PRM*",
            "fmmstar":"FMM*",
            "lazy_prmstar":"Lazy-PRM*",
            "lazy_rrgstar_subopt_0.1":"Lazy LBT-RRG*, eps=0.1",
            "lazy_rrgstar_subopt_0.2":"Lazy LBT-RRG*, eps=0.2",
            "fmtstar":"FMT*",
            #"rrtstar":"RRT*",
            "birrtstar":"RRT*",
            "rrtstar_subopt_0.1":"LBT-RRT*(0.1)",
            "rrtstar_subopt_0.2":"LBT-RRT*(0.2)",
}
#labelorder = ["restart_rrt_shortcut","prmstar","fmmstar","rrtstar","birrtstar","rrtstar_subopt_0.1","rrtstar_subopt_0.2","lazy_prmstar","lazy_rrgstar","lazy_birrgstar"]
labelorder = ["ao_rrt","ao_est","repeated_rrt","repeated_est","repeated_rrt_prune","repeated_est_prune","stable_sparse_rrt","anytime_rrt","rrtstar"]
dashes = [[],[8,8],[4,4],[2,2],[1,1],[12,6],[4,2,2,2],[8,2,2,2,2,2],[6,2],[2,6]]
ylabelmap = {"best cost":"Path length",
             "numEdgeChecks":"# edge checks",
}

timevarname = 'time'
#timevarname = 'numMilestones'
item = sys.argv[2]
with open(sys.argv[1],'r') as f:
    reader = csv.DictReader(f)
    items = defaultdict(list)
    for row in reader:
        time = dict()
        vmean = dict()
        vstd = dict()
        vmin = dict()
        vmax = dict()
        skip = dict()
        for (k,v) in row.iteritems():
            v = float(v) if len(v) > 0 else None
            words = k.split(None,1)
            label = words[0]
            if len(words) >= 2 and words[1] == timevarname:
                time[label] = v
        for (k,v) in row.iteritems():
            v = float(v) if len(v) > 0 else None
            words = k.split(None,1)
            label = words[0]
            if item == 'best cost' and len(words) >= 2 and words[1] == 'success fraction':
                if v < successFraction:
                    skip[label] = True
                else:
                    skip[label] = False
            if len(words) >= 2 and words[1].startswith(item):
                suffix = words[1][len(item)+1:]
                if suffix=='mean': #will have min,max,mean,etc
                    vmean[label] = v
                elif suffix=='std':
                    vstd[label] = v
                elif suffix=='max':
                    vmax[label] = v
                elif suffix=='min':
                    vmin[label] = v
                elif suffix=='':
                    vmean[label] = v
                else:
                    print("Warning, unknown suffix",suffix)
        
        for label,t in time.iteritems():
            if label in skip and skip[label]:
                items[label].append((t,None))
            elif label in vmean:
                items[label].append((t,vmean[label]))
            else:
                print("Warning, no item",item,"for planner",label,"read")
    print("Available planners:",list(items.keys()))

    #small, good for printing
    #fig = plt.figure(figsize=(4,2.7))
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
    if timevarname=='time':
        ax1.set_xlabel("Time (s)")
    else:
        ax1.set_xlabel("Iterations")
    minx = 0
    maxx = 0
    ax1.set_ylabel(ylabelmap.get(item,item))
    for n,label in enumerate(labelorder):
        if label not in items: continue
        plot = items[label]
        if len(items[label])==0:
            print("Skipping item",label
        x,y = zip(*plot)
        minx = min(minx,*[v for v in x if v is not None])
        maxx = max(maxx,*[v for v in x if v is not None])
        plannername = labelmap[label] if label in labelmap else label
        print("Plotting",plannername)
        line = ax1.plot(x,y,label=plannername,dashes=dashes[n])
        plt.setp(line,linewidth=1.5)
    #plt.legend(loc='upper right');
    plt.legend();
    #good for bugtrap cost
    #plt.ylim([2,3])
    #good for other cost
    #plt.ylim([1,2])
    #good for edge checks
    if item=="numEdgeChecks":
        plt.ylim([0,800])
    else:
        #plt.ylim([2,2.8])
        pass
    if timevarname=='time':
        if sys.argv[1].startswith('tx90'):
            plt.xlim([0,20])
        elif sys.argv[1].startswith('baxter'):
            plt.xlim([0,60])
        elif sys.argv[1].startswith('bar_25'):
            plt.xlim([0,20])
        else:
            plt.xlim([math.floor(minx),math.ceil(maxx)])
    else:
        plt.xlim([0,5000])

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend
    #frame = legend.get_frame()
    #frame.set_facecolor('0.97')

    # Set the fontsize
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
    #      ncol=3, fancybox=True, columnspacing=0, handletextpad=0, shadow=False)
    #for label in legend.get_texts():
    #    label.set_fontsize(9)
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0, box.width, box.height* 0.8])
    #plt.setp(ax1.get_xticklabels(),fontsize=12)
    #plt.setp(ax1.get_yticklabels(),fontsize=12)
    #start,end = ax1.get_ylim()
    #ax1.yaxis.set_ticks(np.arange(start, end, 0.1))

    plt.show()
