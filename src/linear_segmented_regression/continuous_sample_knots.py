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

   print("Computing optimal discrete segmented linear fit with %d segments..."%(num_segments))
   
   # Dynamic program to find the optimal discrete linear segmented regression
   n = len(time)
   F = np.nan*np.zeros((n, num_segments))
   I = np.zeros((n, num_segments)).astype(int)
   A = np.nan*np.zeros((n,n))
   B = np.nan*np.zeros((n,n))
   E = np.nan*np.zeros((n,n))
   for i in range(1,n+1):
      E[i-1,i-1] = np.inf
      for j in range(i+1,n+1):
         x = np.array(time.iloc[i-1:j])
         y = np.array(signal.iloc[i-1:j])
         A[i-1,j-1] = ((j-i+1)*np.dot(x,y) - np.sum(x)*np.sum(y))/((j-i+1)*np.dot(x,x) - np.sum(x)**2)
         B[i-1,j-1] = (np.sum(y) - A[i-1,j-1]*np.sum(x))/(j-i+1)
         e = y - A[i-1,j-1]*x - B[i-1,j-1]
         E[i-1,j-1] = np.dot(e,e)

   for j in range(1,n+1):
      for t in range(j,num_segments+1):
         F[j-1,t-1] = np.inf
         I[j-1,t-1] = 0
      F[j-1,0] = E[0,j-1]
      I[j-1,0] = 1
      for t in range(2,min(j,num_segments+1)):
         F[j-1,t-1] = np.inf
         I[j-1,t-1] = 0
         for i in range(t,j+1):
            if F[j-1,t-1] > F[i-1,t-2] + E[i-1,j-1]:
               F[j-1,t-1] = F[i-1,t-2] + E[i-1,j-1]
               I[j-1,t-1] = i

         if show_debug_plots:
            plt.figure()
            plt.plot(signal.index[0:j], signal.iloc[0:j], 'bo')
            i = I[j-1,t-1]
            new_line_x = np.array(signal.index[i-1:j])
            new_line_y = A[i-1,j-1]*new_line_x + B[i-1,j-1]
            plt.plot(new_line_x, new_line_y, 'g--')
            prev_i = I[i-1,t-2]
            right_most_line_x = np.array(signal.index[prev_i-1:i])
            right_most_line_y = A[prev_i-1,i-1]*right_most_line_x + B[prev_i-1,i-1]
            plt.plot(right_most_line_x, right_most_line_y, 'r-')
            cost = E[i-1,j-1]
            plt.title("T=%d, j=%d, Cost of new line: %f"%(t, j, cost))
            plt.show()

   # Recover optimal approximation
   knots = [n]
   i = n
   t = num_segments
   while t > 0:
      knots.append(I[knots[-1]-1,t-1])
      t -= 1
   knots.reverse()
   knots = np.array(knots)
   x = signal.index[knots-1]
   y = [signal.iloc[knots[0]-1]]
   for i in range(1,len(knots)):
      y.append(A[knots[i-1]-1,knots[i]-1]*x[i] + B[knots[i-1]-1,knots[i]-1])

   print("Final approximation loss value: %f"%(F[n-1, num_segments-1]))

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
