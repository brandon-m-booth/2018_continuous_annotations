#!/usr/bin/env python

import os
import sys
import pdb
import argparse
import statprof
import quadprog
import cvxopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

num_segments = 38
#solver = 'quadprog'
solver = 'cvxopt'
do_plot_each_iteration = False

def GetCVXOPTMatrix(M):
   if M is None:
      return None
   elif type(M) is np.ndarray:
      return cvxopt.matrix(M)
   elif type(M) is cvxopt.matrix or type(M) is cvxopt.spmatrix:
      return M
   coo = M.tocoo()
   return cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape)

def GetConstraints(knots, n, start_constant=True):
   G = None
   h = None
   A = np.zeros((n,n))
   b = np.zeros(n)

   # Add a fake knot at the end
   knots.append(n-1)

   last_knot = 0
   for idx in range(len(knots)):
      is_const_constraint = (((idx%2)==0) and start_constant) or (((idx%2)==1) and not start_constant)
      if is_const_constraint:
         interval_points = range(last_knot,knots[idx])
         for x in interval_points:
            A[x,x] = 1
            A[x,x+1] = -1
            b[x] = 0
      else:
         interval_points = range(last_knot,knots[idx]-1)
         for x in interval_points:
            A[x,x] = 1
            A[x,x+1] = -2
            A[x,x+2] = 1
            b[x] = 0
      last_knot = knots[idx]

   zero_row_idx = np.all(A == 0, axis=1)
   A = A[~zero_row_idx]
   b = b[~zero_row_idx]

   return G,h,A,b

def DoOptimalFit(input_csv_path):
   # Get the signal data
   signal_df = pd.read_csv(input_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]

   print("Beginning profile...")
   statprof.start()
   try:
      best_signal = {'loss': sys.float_info.max, 'signal': None}
      n = len(time)
      P = np.eye(n)
      q = np.array(signal) if solver != 'cvxopt' else -np.array(signal)
      knot_indices = range(1,n-1)
      iteration = 0
      for knots in combinations(knot_indices, num_segments):
         knots = list(knots)
         for start_constant in [True, False]:
            G,h,A,b = GetConstraints(knots, n, start_constant)
            if solver == 'quadprog':
               C = A
               meq = A.shape[0]
               signal_fit = quadprog.solve_qp(P, q, C, b, meq)[0]
            elif solver == 'cvxopt':
               P = GetCVXOPTMatrix(P)
               q = GetCVXOPTMatrix(q)
               A = GetCVXOPTMatrix(A)
               b = GetCVXOPTMatrix(b)
               #args = [GetCVXOPTMatrix(P), GetCVXOPTMatrix(q)]
               #args.extend([GetCVXOPTMatrix(G), GetCVXOPTMatrix(h)])
               #args.extend([GetCVXOPTMatrix(A), GetCVXOPTMatrix(b)])
               #cvx_solution = cvxopt.solvers.qp(*args)
               cvx_solution = cvxopt.solvers.qp(P=P, q=q, A=A, b=b)
               if 'optimal' not in cvx_solution['status']:
                  print("Warning: CVXOPT did not find an optimal solution")
               signal_fit = np.array(cvx_solution['x']).reshape((len(q),))
               loss_value = cvx_solution['primal objective']

            if loss_value < best_signal['loss']:
               best_signal['loss'] = loss_value
               best_signal['signal'] = signal_fit
               best_signal['knots'] = knots
               best_signal['iteration'] = iteration

            if do_plot_each_iteration:
               plt.figure()
               plt.plot(time, signal, 'b-')
               plt.plot(time, signal_fit, 'r--')
               plt.show()

         iteration += 1
         print("Finished %d iterations"%(iteration))
         if iteration > 10:
            break
         
   finally:
      statprof.stop()
      statprof.display()
   print("...finished profile")

   print("Best signal knot points: "+str(best_signal['knots']))
   print("Best signal found on iteration: "+str(best_signal['iteration']))

   # Plot results
   plt.figure()
   plt.plot(time, signal, 'b-')
   plt.plot(time, best_signal['signal'], 'r--')
   plt.show()

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
