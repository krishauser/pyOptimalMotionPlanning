# pyOptimalMotionPlanning
Optimal Motion Planning package in Python

v0.1

Kris Hauser

7/27/2015



Requirements:
- Python 2.X
- Numpy/Scipy
- PyOpenGL
- Matplotlib (optional)


Usage:

  "python main.py [-v] [PROBLEM] [PLANNER]"

where -v is an optional flag to show the planner in a GUI.  If not specified,
the planner is run 10 times on the given problem and stats are saved to
disk.  You can also name multiple planners.

[PLANNER] can be any of
 - "r-rrt"
 - "r-est"
 - "r-rrt-prune"
 - "r-est-prune"
 - "ao-rrt"
 - "ao-est"
 - "stable-sparse-rrt" 
 - "sst*" 
 - "all": run all planners
You may also add keyword arguments to change parameters of the planner, e.g.
"r-rrt(numControlSamples=1)".

[PROBLEM] can be any of:
 - "Bugtrap"
 - "Kink"
 - "Pendulum" 
 - "Dubins" 
 - "Dubins2"
 - "DoubleIntegrator"
 - "Flappy"



Visualization controls:

- 'p' will do 1000 iterations of planning
- ' ' will do a single iteration.
- 'm' will start saving frames to disk every 100 iterations of planning, which
  can be later turned into a movie.


Once data has been saved to disk, you can run:

   "python processresults.py [folder]"

to generate a csv file summarizing all of the results for a single
problem.  If you have matplotlib, you can then view the results using

   "python viewsummary [file]"


