#!/usr/bin/env python
#Author: Brandon M. Booth

import os
import sys
import pdb
import math
import argparse
import statprof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib2tikz
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import util

# For debugging
show_final_plot = False
show_debug_plots = False
can_parallelize = False # TODO - Parallelism runs much slower. Investigate and fix me!

# Recovers the optimum TSR of the signal up to index i (1 <= i <= n) for the given number of segments t (1 <= t <= num_segments)
def RecoverOptimumTSR(i, t, X, I, A, B, signal):
   # Find the knot x-axis locations and get the indices of the signal points with x values just before each knot
   knots = [i]
   x = [signal.index[i-1]]
   while t > 0:
      x.append(X[knots[-1]-1,t-1])
      knots.append(I[knots[-1]-1,t-1])
      t -= 1
   knots.reverse()
   x.reverse()

   # Compute the y-axis value at each knot location
   y = [A[knots[1]-1,0]*x[0] + B[knots[1]-1,0]]
   t = 1
   for i in range(1,len(knots)-1):
      y.append(A[knots[i]-1,t-1]*x[i] + B[knots[i]-1,t-1])
      t += 1
   y.append(A[knots[-1]-1,t-1]*x[-1] + B[knots[-1]-1,t-1])

   return (x,y)

def FitNextSegment(signal, n, i, j, t, A, B):
   """
   Helper function to fit the correct next line segment type for trapezoidal functions
   """
   previous_segment_slope = A[i-1,t-2]
   if previous_segment_slope == 0.0:
      (a, b, x, cost) = util.FitLineSegmentWithIntersection(signal, i, j-1, A[i-1,t-2], B[i-1,t-2], max(signal.index[0],signal.index[i-1]), min(signal.index[n-1],signal.index[i]))
   else:
      a = 0
      (b, x, cost) = util.FitConstantSegmentWithIntersection(signal, i, j-1, A[i-1,t-2], B[i-1,t-2], max(signal.index[0],signal.index[i-1]), min(signal.index[n-1],signal.index[i]))
   return a, b, x, cost

def FitNextSegmentStar(args):
   return FitNextSegment(*args)

# Dynamic program to find the optimal trapezoidal segmented regression
def ComputeOptimalFit(input_csv_path, num_segments, max_jobs, output_csv_path, tikz_file=None):
   # Get the signal data
   signal_df = pd.read_csv(input_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   signal.index = time

   print("Computing optimal segmented trapezoidal fit with %d segments..."%(num_segments))
   pool = Pool(max_jobs)
   
   best_x = None
   best_y = None
   best_cost = np.inf
   for start_with_constant_segment in [True, False]:

      n = len(time)
      F = np.nan*np.zeros((n, num_segments))
      I = np.zeros((n, num_segments)).astype(int)
      A = np.nan*np.zeros((n,n))
      B = np.nan*np.zeros((n,n))
      X = np.nan*np.zeros((n, num_segments))
      # Iterate over all sorted points in order
      for j in range(1,n+1):
         print("Computing optimal sub-segmentation for points up to index %d of %d"%(j, n))

         # Initialize costs and set knot indices to an invalid index
         for t in range(j,num_segments+1):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0
            X[j-1,t-1] = 0

         # Fit a single line segment to all points up to the current one
         if start_with_constant_segment:
            a = 0.0
            (b, cost) = util.FitConstantSegment(signal.iloc[0:j])
         else:
            (a,b,cost) = util.FitLineSegment(signal.iloc[0:j])
         A[j-1,0] = a
         B[j-1,0] = b
         X[j-1,0] = signal.index[0]
         F[j-1,0] = cost
         I[j-1,0] = 1

         # Consider using all possible number of segments between the first point and the current
         for t in range(2,min(j,num_segments+1)):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0
            X[j-1,0] = 0

            # For the target number of t segments, find the best break point reusing the optimum
            # fit for t-1 segments over all points up to some point before the current
            last_knots = []
            for i in range(t,j):
               k = I[i-1,t-2]
               if k != 0 and A[k-1,i-1] != A[i-1,j-1]:
                  last_knots.append(i)

            next_segment_args = [(signal, n, i, j, t, A, B) for i in last_knots]
            if can_parallelize:
               results = pool.map(FitNextSegmentStar, next_segment_args)
            else:
               results = [FitNextSegmentStar(params) for params in next_segment_args]

            # Update the cost, linear coefficients, and break points
            avals, bvals, xvals, costs = zip(*results)
            last_knots = np.array(last_knots)
            prev_sum_costs = F[last_knots-1,t-2]
            min_idx = np.argmin(prev_sum_costs+costs)
            if F[j-1,t-1] > (prev_sum_costs+costs)[min_idx]:
               F[j-1,t-1] = (prev_sum_costs+costs)[min_idx]
               A[j-1,t-1] = avals[min_idx]
               B[j-1,t-1] = bvals[min_idx]
               I[j-1,t-1] = last_knots[min_idx]
               X[j-1,t-1] = xvals[min_idx]

            if show_debug_plots:
               for results_idx in range(len(results)):
                  a,b,x,cost = results[results_idx]
                  i = last_knots[results_idx]
                  if not np.isinf(cost):
                     # Get best TSR up to signal index i for t-1 segments
                     best_tsr_so_far_x, best_tsr_so_far_y = RecoverOptimumTSR(i, t-1, X, I, A, B, signal)

                     # Store the best line segment computed this iteration
                     new_line_x = np.array(signal.index[i-1:j])
                     new_line_y = a*new_line_x + b

                     # Find the intersection of the new line and the best TSR so far
                     c = (best_tsr_so_far_y[-2]-b-a*best_tsr_so_far_x[-2])/(a*(best_tsr_so_far_x[-1]-best_tsr_so_far_x[-2])-best_tsr_so_far_y[-1]+best_tsr_so_far_y[-2])
                     x_int = best_tsr_so_far_x[-2] + c*(best_tsr_so_far_x[-1]-best_tsr_so_far_x[-2])
                     y_int = best_tsr_so_far_y[-2] + c*(best_tsr_so_far_y[-1]-best_tsr_so_far_y[-2])

                     # Fix up the TSR and new line points so they share the intersection
                     best_tsr_so_far_x[-1] = x_int
                     best_tsr_so_far_y[-1] = y_int
                     new_line_x[0] = x_int
                     new_line_y[0] = y_int

                     plt.figure()
                     plt.plot(signal.index[0:i], signal.iloc[0:i], 'mo')
                     plt.plot(signal.index[i:j], signal.iloc[i:j], 'bo')
                     plt.plot(new_line_x, new_line_y, 'g--')
                     plt.plot(best_tsr_so_far_x, best_tsr_so_far_y, 'r-')
                     plt.title("T=%d, i=%d, j=%d, Cost of new line: %f"%(t, i, j, cost))
                     plt.show()

      # Recover optimum TSR
      x,y = RecoverOptimumTSR(n, num_segments, X, I, A, B, signal)

      start_segment_type = "constant-segment-first" if start_with_constant_segment else "linear-segment-first"
      print("Final %s approximation loss value: %f"%(start_segment_type, F[n-1, num_segments-1]))

      cost = F[n-1, num_segments-1]
      if cost < best_cost:
         best_x = x
         best_y = y

   out_df = pd.DataFrame(data={'Time': best_x, 'Value': best_y})
   out_df.to_csv(output_csv_path, header=True, index=False)

   # Plot results
   plt.figure()
   plt.plot(time, signal, 'bo')
   plt.plot(best_x, best_y, 'r-')
   plt.title("Best fit with MSE cost: %f"%(best_cost))
   if tikz_file is not None:
      matplotlib2tikz.save(tikz_file)
   if show_final_plot:
      plt.show()
      

   return


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='CSV-formatted input signal file with (first column: time, second: signal value)')
   parser.add_argument('--segments', dest='num_segments', required=True, help='Number of segments to use in the approximation')
   parser.add_argument('--maxjobs', dest='max_jobs', required=False, help='The maximum number of parallel jobs allowed.  For maximal efficiency, set this value to the number of processing cores available')
   parser.add_argument('--output', dest='output_csv', required=True, help='Output csv path')
   parser.add_argument('--tikz', dest='tikz', required=False, help='Output path for TikZ PGF plot code') 
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_path = args.input_csv
   num_segments = int(args.num_segments)
   max_jobs = int(args.max_jobs) if args.max_jobs is not None else 1
   tikz_file = args.tikz
   output_csv_path = args.output_csv
   ComputeOptimalFit(input_csv_path, num_segments, max_jobs, output_csv_path, tikz_file=tikz_file)
