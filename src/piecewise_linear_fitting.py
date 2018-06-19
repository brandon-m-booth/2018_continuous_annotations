import os
import sys
import pdb
import numpy as np
import pandas as pd
import statprof
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

def GetElbowIndices(signal, num_elbows):
   elbow_indices = []
   d2 = np.diff(np.diff(signal))
   for i in range(num_elbows): # BB - np.argsort doesn't work!?
      argmax = np.nanargmax(np.abs(d2))
      d2[argmax] = 0
      elbow_indices.append(argmax)
   return elbow_indices

def GetConstantIntervals(signal):
   intervals = []
   last_val = signal[0]
   interval_start_idx = -1
   for idx in range(1, len(signal)):
      if signal[idx] == last_val:
         if interval_start_idx < 0:
            interval_start_idx = idx-1
         else:
            pass # Keep searching along this interval...
      else:
         if interval_start_idx < 0:
            pass # Keep searching for the next interval...
         else:
            intervals.append((interval_start_idx, idx-1))
            interval_start_idx = -1
      last_val = signal[idx]

   if interval_start_idx >= 0:
      intervals.append((interval_start_idx, len(signal)-1))

   return intervals

# Computes the optimal warped version of the signal where:
# 1) The constant intervals have the same domain and are adjusted to the mean of the signal (the constraint)
# 2) Lines are used to connect the adjusted (locally optimal) constant intervals
def ComputeOptimalConstrainedLinearFit(signal, intervals):
   opt_signal = np.zeros_like(signal)

   # Adjust constant intervals to the signal mean
   for interval in intervals:
      m = np.mean(signal[interval[0]:interval[1]+1])
      opt_signal[interval[0]:interval[1]+1] = m

   # Connect the intervals with lines
   interval_idx = 0
   left_idx = 0
   right_idx = intervals[interval_idx][0]
   while True:
      left_val = signal[left_idx]
      right_val = signal[right_idx]
      lerp = np.interp(np.arange(left_idx+1,right_idx), [left_idx, right_idx], [left_val, right_val])
      opt_signal[left_idx+1:right_idx] = lerp

      interval_idx += 1
      if interval_idx < len(intervals):
         left_idx = intervals[interval_idx-1][1]
         right_idx = intervals[interval_idx][0]
      elif interval_idx == len(intervals)+1:
         left_idx = intervals[interval_idx-1][1]
         right_idx = len(signal)-1
      else:
         break

   return opt_signal
      
def DoPiecewiseLinearFitting(signal_csv_path):
   signal_df = pd.read_csv(signal_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   original_signal = signal.copy()

   plt.figure()

   # For optimal signal
   fit_signal_dict = {}

   # For plots
   constant_intervals_per_iter = []
   total_interval_length_per_iter = []
   optimal_constrained_mse = []


   iteration = 0
   delta = np.diff(signal)
   is_finished = np.sum(delta) == 0
   while not is_finished:
      delta[delta == 0] = np.nan
      min_idx = np.nanargmin(np.abs(delta))

      left_idx = min_idx
      right_idx = min_idx + 1
      left_diff = 0
      right_diff = 0
      if left_idx - 1 > 0:
         left_diff = abs(signal[left_idx - 1] - signal[left_idx])
      if right_idx + 1 < len(signal):
         right_diff = abs(signal[right_idx] - signal[right_idx + 1])

      if left_diff == 0 and right_diff == 0:
         # Merge the two intervals keeping the longer one intact
         left_edge_idx = left_idx
         while left_edge_idx >= 0 and signal[left_edge_idx] == signal[left_idx]:
            left_edge_idx = left_edge_idx - 1
         left_edge_idx = left_edge_idx + 1

         right_edge_idx = right_idx
         while right_edge_idx < len(signal) and signal[right_edge_idx] == signal[right_idx]:
            right_edge_idx = right_edge_idx + 1
         right_edge_idx = right_edge_idx - 1

         left_interval_len = left_idx - left_edge_idx + 1
         right_interval_len = right_edge_idx - right_idx + 1
         if left_interval_len < right_interval_len:
            signal[left_edge_idx:left_idx+1] = signal[right_idx]
         else:
            signal[right_idx:right_edge_idx+1] = signal[left_idx]
      elif left_diff == 0:
         signal[right_idx] = signal[left_idx]
      elif right_diff == 0:
         signal[left_idx] = signal[right_idx]
      else:
         # Pick the side with the smallest diff
         if left_diff < right_diff:
            signal[right_idx] = signal[left_idx]
         else:
            signal[left_idx] = signal[right_idx]

      # Get the constant intervals
      intervals = GetConstantIntervals(signal)
      constant_intervals_per_iter.append(len(intervals))

      # Compute the length of each constant intervals
      interval_lengths = []
      for interval in intervals:
         interval_lengths.append(interval[1]-interval[0] + 1)
      total_interval_length_per_iter.append(np.sum(interval_lengths))

      # Compute the best MSE given the current set of constant intervals
      do_compute_mse = False
      if total_interval_length_per_iter[-1] == len(signal):
         do_compute_mse = True

      if do_compute_mse:
         opt_signal = ComputeOptimalConstrainedLinearFit(original_signal, intervals)
         opt_mse = mean_squared_error(original_signal, opt_signal)
         optimal_constrained_mse.append(opt_mse)
      else:
         optimal_constrained_mse.append(np.nan)
         opt_signal = None

      # Plot the progress
      do_plot = do_compute_mse
      if do_plot:
         plt.clf()

         plt.subplot(321)
         plt.plot(time, signal)
         plt.plot(time[left_idx], signal[left_idx], 'o')
         plt.title('Iteration ' + str(iteration))

         plt.subplot(322)
         plt.plot(constant_intervals_per_iter)
         plt.title('Number constant intervals per iteration')

         plt.subplot(323)
         plt.plot(total_interval_length_per_iter)
         plt.title('Total constant intervals length per iteration')

         plt.subplot(324)
         plt.hist(interval_lengths)
         plt.title('Distribution of interval lengths')

         plt.subplot(325)
         plt.plot(optimal_constrained_mse)
         plt.title('Optimal Constrained MSE')

         plt.subplot(326)
         plt.plot(time, opt_signal)
         plt.plot(time, original_signal, 'r-')
         plt.title('Optimal Constrained Signal')

         plt.draw()
         plt.pause(0.001)
      else:
         print('Iteration: %d'%(iteration))

      fit_signal_dict[iteration] = {'iteration_signal': signal.copy(), 'opt_fit_signal': opt_signal}

      #if iteration == 2077:
      #   pdb.set_trace()

      delta = np.diff(signal)
      is_finished = np.sum(delta) == 0
      iteration = iteration + 1

   # Find and plot the elbow joints in the optimal constrained MSE plot
   elbow_indices = GetElbowIndices(optimal_constrained_mse, 3)
   plt.subplot(326)
   plt.plot(elbow_indices, np.array(optimal_constrained_mse)[elbow_indices], 'ro')
   plt.draw()
   plt.pause(0.001)
   for elbow_index in elbow_indices:
      iter_signal = fit_signal_dict[elbow_index]['iteration_signal']
      fit_signal = fit_signal_dict[elbow_index]['opt_fit_signal']

      plt.figure()
      plt.plot(time, fit_signal, 'b')
      plt.plot(time, original_signal, 'r-')
      plt.plot(time, iter_signal, 'g--')
      plt.legend(['Optimal', 'Original', 'Iteration'])
      plt.title('Optimal constrained linear fit: iteration %d'%(elbow_index))
      plt.draw()
      plt.pause(0.001)

   pdb.set_trace()
   return fit_signal

if __name__ == '__main__':
   if len(sys.argv) > 2:
      signal_csv_path = sys.argv[1]
      statprof.start()
      try:
         fit_signal = DoPiecewiseLinearFitting(signal_csv_path)
      finally:
         statprof.stop()
         statprof.display()
   else:
      print("Please provide the following command line arguments:\n 1) Path to signal csv file\n 2) Desired number of linear segments")
