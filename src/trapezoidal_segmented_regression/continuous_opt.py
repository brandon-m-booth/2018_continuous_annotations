#!/usr/bin/env python

import os
import sys
import pdb
import math
import argparse
import statprof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import util

# For debugging
show_debug_plots = False

def ComputeOptimalFit(input_csv_path, num_segments):
   # Get the signal data
   signal_df = pd.read_csv(input_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   signal.index = time

   print("Computing optimal segmented trapezoidal fit with %d segments..."%(num_segments))
   
   for start_with_constant_segment in [True, False]:
      # Dynamic program to find the optimal trapezoidal segmented regression
      n = len(time)
      F = np.nan*np.zeros((n, num_segments))
      I = np.zeros((n, num_segments)).astype(int)
      A = np.nan*np.zeros((n,n))
      B = np.nan*np.zeros((n,n))
      X = np.nan*np.zeros((n, num_segments))
      for j in range(1,n+1):
         print("Computing optimal sub-segmentation for points up to index %d of %d"%(j, n))
         for t in range(j,num_segments+1):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0
            X[j-1,t-1] = 0
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
         for t in range(2,min(j,num_segments+1)):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0
            X[j-1,0] = 0
            for i in range(t,j):
               k = I[i-1,t-2]
               if k != 0 and A[k-1,i-1] != A[i-1,j-1]:
                  previous_segment_slope = A[i-1,t-2]
                  if previous_segment_slope == 0.0:
                     (a, b, x, cost) = util.FitLineSegmentWithIntersection(signal, i, j-1, A[i-1,t-2], B[i-1,t-2], max(0,signal.index[i-1]), min(signal.index[n-1],signal.index[i]))
                  else:
                     a = 0
                     (b, x, cost) = util.FitConstantSegmentWithIntersection(signal, i, j-1, A[i-1,t-2], B[i-1,t-2], max(0,signal.index[i-1]), min(signal.index[n-1],signal.index[i]))

                  if show_debug_plots and not np.isinf(cost):
                     plt.figure()
                     plt.plot(signal.index[0:j], signal.iloc[0:j], 'bo')
                     new_line_x = np.array(signal.index[i-1:j])
                     new_line_y = a*new_line_x + b
                     plt.plot(new_line_x, new_line_y, 'g--')
                     right_most_line_x = np.array(signal.index[0:i+1])
                     right_most_line_y = A[i-1,t-2]*right_most_line_x + B[i-1,t-2]
                     plt.plot(right_most_line_x, right_most_line_y, 'r-')
                     plt.title("T=%d, i=%d, j=%d, Cost of new line: %f"%(t, i, j, cost))
                     plt.show()

                  if F[j-1,t-1] > F[i-1,t-2] + cost:
                     F[j-1,t-1] = F[i-1,t-2] + cost
                     A[j-1,t-1] = a
                     B[j-1,t-1] = b
                     I[j-1,t-1] = i
                     X[j-1,t-1] = x

      # Recover optimal approximation
      knots = [n]
      x = [signal.index[n-1]]
      i = n
      t = num_segments
      while t > 0:
         x.append(X[knots[-1]-1,t-1])
         knots.append(I[knots[-1]-1,t-1])
         t -= 1
      knots.reverse()
      x.reverse()
      y = [A[knots[1]-1,0]*x[0] + B[knots[1]-1,0]]
      t = 1
      for i in range(1,len(knots)-1):
         y.append(A[knots[i]-1,t-1]*x[i] + B[knots[i]-1,t-1])
         t += 1
      y.append(A[knots[-1]-1,t-1]*x[-1] + B[knots[-1]-1,t-1])

      start_segment_type = "constant-segment-first" if start_with_constant_segment else "linear-segment-first"
      print("Final %s approximation loss value: %f"%(start_segment_type, F[n-1, num_segments-1]))

      # Plot results
      plt.figure()
      plt.plot(time, signal, 'bo')
      plt.plot(x, y, 'r-')
      plt.show()

   return


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='CSV-formatted input signal file with (first column: time, second: signal value)')
   parser.add_argument('--segments', dest='num_segments', required=True, help='Number of segments to use in the approximation')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_path = args.input_csv
   num_segments = int(args.num_segments)
   ComputeOptimalFit(input_csv_path, num_segments)
