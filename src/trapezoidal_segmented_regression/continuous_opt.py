#!/usr/bin/env python

import os
import sys
import pdb
import math
import argparse
import statprof
import quadprog
import cvxopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

num_segments = 9
show_debug_plots = False

def GetCVXOPTMatrix(M):
   if M is None:
      return None
   elif type(M) is np.ndarray:
      return cvxopt.matrix(M)
   elif type(M) is cvxopt.matrix or type(M) is cvxopt.spmatrix:
      return M
   coo = M.tocoo()
   return cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape)

def FitConstantSegment(signal):
   if len(signal) > 1:
      b = np.mean(signal.iloc[1:])
      residuals = signal.iloc[1:] - b*np.ones(len(signal)-1)
      loss_value = np.dot(residuals,residuals)
   elif len(signal) == 1:
      b = signal.iloc[0]
      loss_value = 0.0
   else:
      b = np.nan
      loss_value = np.inf
   return b, loss_value

def FitLineSegment(signal):
   if len(signal) > 0:
      try:
         M = np.vstack((signal.index, np.ones(len(signal)))).T
         coefs, residuals, rank, singular_vals = np.linalg.lstsq(M, signal.values, rcond=None)
         a,b = coefs
         if len(residuals) > 0:
            average_error = float(residuals)/len(signal)
         else:
            average_error = 0.0
      except:
         a = 0
         b = signal.iloc[0]
         average_error = float(np.sum(np.abs(np.array(signal)-b)))/len(signal)
   else:
      a = np.nan
      b = np.nan
      average_error = np.inf
   return a, b, average_error

def FitConstantSegmentWithIntersection(signal, i, j, a, b, x1, x2):
   if np.isnan(a) or np.isnan(b):
      return np.nan, np.nan, np.inf

   x = np.array(signal.iloc[i:j+1].index)
   y = np.array(signal.iloc[i:j+1])
   P = np.array([len(x)]).astype(float)
   q = np.array([-np.sum(y)]).astype(float)
   G = np.array([[1], [-1]]).astype(float)
   h = np.array([a*x2+b, -a*x1-b]).reshape(-1,1).astype(float)
   if a < 0:
      G = -G
      h = -h
   
   P = GetCVXOPTMatrix(P)
   q = GetCVXOPTMatrix(q)
   G = GetCVXOPTMatrix(G)
   h = GetCVXOPTMatrix(h)
   cvxopt.solvers.options['maxiters'] = 300
   cvx_solution = cvxopt.solvers.coneqp(P=P, q=q, G=G, h=h, kktsolver='ldl')
   loss_value = np.nan
   if 'optimal' in cvx_solution['status']:
      loss_value = cvx_solution['primal objective']
      signal_fit = np.array(cvx_solution['x']).reshape((len(q),))
   else:
      print("Warning: CVXOPT did not find an optimal solution")

   loss_value += 0.5*np.dot(y,y)
   if loss_value < 0: # Handle numerical precision issues
      loss_value = 0
   if a == 0.0:
      x = (x1+x2)/2.0
   else:
      x = (signal_fit[0]-b)/a
   return signal_fit[0], x, loss_value

def FitLineSegmentWithIntersection(signal, i, j, a, b, x1, x2):
   if np.isnan(a) or np.isnan(b):
      return np.nan, np.nan, np.nan, np.inf

   # Special case: if signal.iloc[i:j+1] has one point, then there is no unique optimal
   # solution. Pick the line going through a*x1+b and signal[i].
   if i == j:
      x_point = x1
      new_a = (signal.iloc[i]-(a*x_point+b))/(signal.index[i]-x_point)
      new_b = signal.iloc[i] - new_a*signal.index[i]
      loss_value = 0.0
      return new_a, new_b, x_point, loss_value

   x = np.array(signal.index[i:j+1])
   y = np.array(signal.iloc[i:j+1])
   x_squared = np.square(x)
   P = np.array([[np.sum(x_squared), np.sum(x)],[np.sum(x), len(x)]]).astype(float)
   q = np.array([-np.dot(x,y), -np.sum(y)]).astype(float)
   G1 = np.array([[-x2, -1], [x1, 1], [-1, 0]]).astype(float)
   h1 = np.array([-a*x2-b, a*x1+b, -a]).reshape(-1,1).astype(float)
   G2 = -G1
   h2 = -h1
   
   P = GetCVXOPTMatrix(P)
   q = GetCVXOPTMatrix(q)
   G1 = GetCVXOPTMatrix(G1)
   h1 = GetCVXOPTMatrix(h1)
   G2 = GetCVXOPTMatrix(G2)
   h2 = GetCVXOPTMatrix(h2)
   cvxopt.solvers.options['maxiters'] = 300
   cvx_solution1 = cvxopt.solvers.coneqp(P=P, q=q, G=G1, h=h1, kktsolver='ldl')
   cvx_solution2 = cvxopt.solvers.coneqp(P=P, q=q, G=G2, h=h2, kktsolver='ldl')
   loss_value1 = cvx_solution1['primal objective']
   loss_value2 = cvx_solution2['primal objective']
   if 'optimal' in cvx_solution1['status']:
      if 'optimal' in cvx_solution2['status'] and loss_value2 < loss_value1:
         loss_value = loss_value2
         signal_fit = np.array(cvx_solution2['x']).reshape((len(q),))
      else:
         loss_value = loss_value1
         signal_fit = np.array(cvx_solution1['x']).reshape((len(q),))
   elif 'optimal' in cvx_solution2['status']:
      loss_value = loss_value2
      signal_fit = np.array(cvx_solution2['x']).reshape((len(q),))
   else:
      print("Warning: CVXOPT did not find an optimal solution")
      loss_value = None

   loss_value += 0.5*np.dot(y,y)
   if loss_value < 0: # Handle numerical precision issues
      loss_value = 0
   if abs(a-signal_fit[0]) < 0.00001:
      x = (x1+x2)/2.0 # If the lines are parallel, use a valid x value
   else:
      x = (signal_fit[1]-b)/(a-signal_fit[0])
   return signal_fit[0], signal_fit[1], x, loss_value

def DoOptimalFit(input_csv_path):
   # Get the signal data
   signal_df = pd.read_csv(input_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   signal.index = time

   print("Beginning profile...")
   statprof.start()
   try:
      # Dynamic program to find the optimal trapezoidal segmented regression
      n = len(time)
      F = np.nan*np.zeros((n, num_segments))
      I = np.zeros((n, num_segments)).astype(int)
      A = np.nan*np.zeros((n,n))
      B = np.nan*np.zeros((n,n))
      X = np.nan*np.zeros((n, num_segments))
      for j in range(1,n+1):
         for t in range(j,num_segments+1):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0
            X[j-1,t-1] = 0
         (a,b,cost) = FitLineSegment(signal.iloc[0:j])
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
                     (a, b, x, cost) = FitLineSegmentWithIntersection(signal, i, j-1, A[i-1,t-2], B[i-1,t-2], max(0,signal.index[i-1]), min(signal.index[n-1],signal.index[i]))
                  else:
                     a = 0
                     (b, x, cost) = FitConstantSegmentWithIntersection(signal, i, j-1, A[i-1,t-2], B[i-1,t-2], max(0,signal.index[i-1]), min(signal.index[n-1],signal.index[i]))

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

      # Plot results
      plt.figure()
      plt.plot(time, signal, 'bo')
      plt.plot(x, y, 'r-')
      plt.show()

   finally:
      statprof.stop()
      statprof.display()
   print("...finished profile")


   return


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='CSV-formatted input signal file with (first column: time, second: signal value)')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_path = args.input_csv
   DoOptimalFit(input_csv_path)
