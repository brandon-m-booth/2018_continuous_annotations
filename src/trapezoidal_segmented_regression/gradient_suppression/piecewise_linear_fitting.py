import os
import sys
import pdb
import numpy as np
import pandas as pd
import statprof
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

CONST_VAL = -999999 # Some constant value out of the signal's normal range
require_plot_input = False

def GetElbowIndices(signal, num_elbows):
   elbow_indices = []
   d2 = np.diff(np.diff(signal))
   for i in range(num_elbows): # BB - np.argsort doesn't work!?
      argmax = np.nanargmax(np.abs(d2))
      d2[argmax] = 0
      elbow_indices.append(argmax)
   return elbow_indices

def GetConstantIntervals(signal, const_value=None):
   intervals = []
   last_val = signal[0]
   interval_start_idx = -1
   for idx in range(1, len(signal)):
      if signal[idx] == last_val and (const_value is None or signal[idx] == const_value):
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
   opt_signal = signal.copy()

   # Adjust constant intervals to the signal mean
   for interval in intervals:
      m = np.mean(signal.loc[interval[0]:interval[1]])
      opt_signal.loc[interval[0]:interval[1]] = m

   # Connect the intervals with lines
   interval_idx = 0
   left_idx = signal.index[0]
   right_idx = opt_signal.index[-1]
   if len(intervals) > 0:
      right_idx = intervals[interval_idx][0]
   while True:
      if left_idx < right_idx:
         left_val = opt_signal[left_idx]
         right_val = opt_signal[right_idx]
         lerp = np.interp(np.arange(left_idx,right_idx+1), [left_idx, right_idx], [left_val, right_val])
         opt_signal.loc[left_idx:right_idx] = lerp

      interval_idx += 1
      if interval_idx < len(intervals):
         left_idx = intervals[interval_idx-1][1]
         right_idx = intervals[interval_idx][0]
      elif interval_idx == len(intervals):
         left_idx = intervals[interval_idx-1][1]
         right_idx = signal.index[-1]
      else:
         break

   return opt_signal
 
def GetIntraIntervals(signal, intervals):
   intra_intervals = []
   last_right_idx = -1
   for interval in intervals:
      if interval[0] > 0:
         intra_intervals.append([last_right_idx+1, interval[0]-1])
      last_right_idx = interval[1]
   if last_right_idx < len(signal)-1:
      intra_intervals.append([last_right_idx+1, len(signal)-1])

   return intra_intervals

def ComputeIntervalActionsMSE(signal, intervals, original_signal):
   interval_actions_mse = {}
   for i in range(len(intervals)):
      interval = intervals[i]
      interval_actions_mse[i] = []

      if i > 0:
         prev_interval = intervals[i-1]
      else:
         prev_interval = [0,0]
      if i > 1:
         prev_prev_interval = intervals[i-2]
      else:
         prev_prev_interval = [0,0]
      if i < len(intervals)-1:
         next_interval = intervals[i+1]
      else:
         next_interval = [len(signal)-1, len(signal)-1]
      if i < len(intervals)-2:
         next_next_interval = intervals[i+2]
      else:
         next_next_interval = [len(signal)-1, len(signal)-1]

      actions = ['remove', 'merge_left', 'merge_right', 'merge_both']
      for action in actions:
         # Check the effect on MSE for each action
         if action == 'remove':
            l = prev_interval[1]
            r = next_interval[0]+1
            ll = l
            rr = r
            #dummy_intervals = [(0,0), (r-1,r-1)]
            dummy_intervals = []
         elif action == 'merge_left':
            ll = prev_prev_interval[1]
            l = prev_interval[0]
            r = interval[1]+1
            rr = next_interval[0]+1
            dummy_intervals = [(l,r-1)]
         elif action == 'merge_right':
            ll = prev_interval[1]
            l = interval[0]
            r = next_interval[1]+1
            rr = next_next_interval[0]+1
            dummy_intervals = [(l,r-1)]
         elif action == 'merge_both':
            ll = prev_prev_interval[1]
            l = prev_interval[0]
            r = next_interval[1]+1
            rr = next_next_interval[0]+1
            dummy_intervals = [(l,r-1)]

         # Handle edge cases
         if (action == 'merge_left' or action == 'merge_both') and l <= 0:
            continue
         if (action == 'merge_right' or action == 'merge_both') and r >= len(signal):
            continue

         orig_sig_with_boundaries = original_signal.copy()
         orig_sig_with_boundaries[ll] = signal[ll]
         orig_sig_with_boundaries[rr-1] = signal[rr-1]
         new_signal = ComputeOptimalConstrainedLinearFit(orig_sig_with_boundaries[ll:rr], dummy_intervals)
         current_mse = (rr-ll)*mean_squared_error(original_signal[ll:rr], signal[ll:rr])
         new_mse = (rr-ll)*mean_squared_error(original_signal[ll:rr], new_signal)
         delta_mse = new_mse-current_mse
         interval_actions_mse[i].append((action, delta_mse))

   return interval_actions_mse


def ComputeIntraIntervalMSE(signal, intervals, original_signal):
   intra_intervals = GetIntraIntervals(signal, intervals)
   #opt_signal = ComputeOptimalConstrainedLinearFit(original_signal, intervals)

   out_intra_interval_mses = []
   for intra_interval in intra_intervals:
      l = intra_interval[0]
      r = intra_interval[1]+1
      mse = (r-l)*mean_squared_error(original_signal[l:r], signal[l:r])
      out_intra_interval_mses.append((intra_interval, mse))

   return out_intra_interval_mses

def CountConvexIntraIntervals(signal, intervals):
   intra_intervals = GetIntraIntervals(signal, intervals)

   num_convex = 0
   num_total = 0
   for intra_interval in intra_intervals:
      intra_interval_signal = signal[intra_interval[0]:intra_interval[1]]
      hessian = np.diff(np.diff(intra_interval_signal))
      if np.abs(np.sum(np.sign(hessian))) == len(intra_interval_signal):
         num_convex += 1
      num_total += 1

   return num_convex, num_total
      
def DoPiecewiseLinearFitting(signal_csv_path, out_pickle_file):
   signal_df = pd.read_csv(signal_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   original_signal = signal.copy()

   plt.figure()

   # For optimal signal
   fit_signal_dict = {'original_signal': original_signal}

   # For plots
   constant_intervals_per_iter = []
   total_interval_length_per_iter = []
   optimal_constrained_mse = []
   num_convex_intra_intervals = []

   delta = np.diff(signal)

   iteration = 0
   is_finished = False
   while not is_finished:
      strategy = 'worst_intra_interval'
      if strategy == 'min_gradient':
         if total_interval_length_per_iter and total_interval_length_per_iter[-1] == len(signal):
            do_compute_mse = True
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

         delta = np.diff(signal)
         is_finished = np.sum(delta) == 0
         get_intervals_const = None
      elif strategy == 'min_area': # TODO
         do_compute_mse = True
         min_idx = np.nanargmin(np.abs(delta))
         get_intervals_const = None
      elif strategy == 'gradient_threshold' or strategy == 'worst_intra_interval':
         do_compute_mse = True #iteration > 300

         if strategy == 'gradient_threshold' or iteration == 0:
            if 'sorted_abs_delta' not in locals():
               sorted_abs_delta = np.unique(np.sort(np.abs(delta)))
            threshold_indices = np.where(np.abs(delta) <= sorted_abs_delta[iteration])[0].tolist()
            threshold_indices.extend(np.array(threshold_indices)+1)
            threshold_indicies = np.unique(threshold_indices)
            do_threshold_add_interval = True
            get_intervals_const = None

         else: # strategy == 'worst_intra_interval' and iteration > 0
            if 'phase_two' not in locals():
               phase_two = False

            if not phase_two:
               intra_interval_mse = ComputeIntraIntervalMSE(opt_signal, intervals, original_signal)
               max_idx = np.argmax(zip(*intra_interval_mse)[1])
               intra_interval = intra_interval_mse[max_idx][0]
               intra_interval_abs_delta = np.abs(np.diff(signal[intra_interval[0]:intra_interval[1]+1]))

               # Stopping condition: intra_interval is too small
               if intra_interval[1]-intra_interval[0] <= 0:
                  phase_two = True
               else:
                  threshold = np.sort(intra_interval_abs_delta)[0]
                  threshold_indices = np.where(intra_interval_abs_delta <= threshold)[0].tolist()
                  threshold_indices.extend(np.array(threshold_indices)+1)
                  threshold_indices = np.unique(threshold_indices)
                  threshold_indices += intra_interval[0]
               do_threshold_add_interval = True
               get_intervals_const = CONST_VAL
            else:
               # Find the best action for some interval (remove, merge left, merge right, merge both sides)
               delta_mse_for_interval_actions = ComputeIntervalActionsMSE(opt_signal, intervals, original_signal)
               min_delta_mse = sys.float_info.max
               best_action = ''
               best_interval = []
               for interval_idx in delta_mse_for_interval_actions.keys():
                  actions, delta_mse = zip(*delta_mse_for_interval_actions[interval_idx])
                  for i in range(len(delta_mse)):
                     if delta_mse[i] < min_delta_mse:
                        min_delta_mse = delta_mse[i]
                        best_action = actions[i]
                        best_interval_idx = interval_idx

               # Get adjacent intervals
               interval = intervals[best_interval_idx]
               if best_interval_idx > 0:
                  prev_interval = intervals[best_interval_idx-1]
               else:
                  prev_interval = [0,0]
               if best_interval_idx < len(intervals)-1:
                  next_interval = intervals[best_interval_idx+1]
               else:
                  next_interval = [len(intervals)-1, len(intervals)-1]

               # Perform the best action
               if best_action == 'remove':
                  threshold_indices = range(interval[0], interval[1]+1)
                  do_threshold_add_interval = False
               elif best_action == 'merge_left':
                  threshold_indices = range(prev_interval[0], interval[1]+1)
                  do_threshold_add_interval = True
               elif best_action == 'merge_right':
                  threshold_indices = range(interval[0], next_interval[1]+1)
                  do_threshold_add_interval = True
               elif best_action == 'merge_both':
                  threshold_indices = range(prev_interval[0], next_interval[1]+1)
                  do_threshold_add_interval = True
               else:
                  print "Unknown best action. Fix me!"
                  pdb.set_trace()
               get_intervals_const = CONST_VAL
                  

         if 'constant_indices' not in locals():
            constant_indices = np.zeros_like(original_signal).astype(bool)
         constant_indices[threshold_indices] = do_threshold_add_interval
         signal = original_signal.copy()
         signal[constant_indices] = CONST_VAL

         if 'last_signal' in locals():
            new_constant_points = np.where(signal-last_signal != 0)[0]
         last_signal = signal
         if strategy != 'worst_intra_interval':
            is_finished = iteration == len(sorted_abs_delta)-1


      # Get the constant intervals
      intervals = GetConstantIntervals(signal, const_value=get_intervals_const)
      constant_intervals_per_iter.append(len(intervals))

      # Compute the length of each constant intervals
      interval_lengths = []
      for interval in intervals:
         interval_lengths.append(interval[1]-interval[0] + 1)
      total_interval_length_per_iter.append(np.sum(interval_lengths))

      if do_compute_mse:
         opt_signal = ComputeOptimalConstrainedLinearFit(original_signal, intervals)
         opt_mse = len(opt_signal)*mean_squared_error(original_signal, opt_signal)
         optimal_constrained_mse.append(opt_mse)
         if strategy == 'gradient_threshold':
            # Count convex intra-intervals
            num_convex, num_total = CountConvexIntraIntervals(original_signal, intervals)
            num_convex_intra_intervals.append((num_convex, num_total))

            # Figure out where the new intervals are in relation to last iteration
            if 'new_constant_points' in locals():
               new_intervals = []
               for new_constant_point in new_constant_points:
                  if len(new_intervals) == 0:
                     new_intervals.append([new_constant_point, new_constant_point])
                  else:
                     if new_constant_point == new_intervals[-1][1]+1:
                        new_intervals[-1][1] = new_constant_point
                     else:
                        new_intervals.append([new_constant_point, new_constant_point])
               new_constant_interval_locations = []
               for new_interval in new_intervals:
                  new_location = 'split'
                  if new_interval[0] > 0:
                     if signal[new_interval[0]-1] == CONST_VAL:
                        new_location = 'extend'
                  if new_interval[1] < len(signal)-1:
                     if signal[new_interval[1]+1] == CONST_VAL:
                        if new_location == 'extend':
                           new_location = 'collapse'
                        else:
                           new_location = 'extend'
                           
                  new_constant_interval_locations.append(new_location)
               if len(new_constant_interval_locations) == 0:
                  new_constant_interval_locations.append('none (collapsed last iter)')
               if len(optimal_constrained_mse) > 1:
                  delta_mse = opt_mse - optimal_constrained_mse[-2]
               else:
                  delta_mse = 0
               print 'Iteration %d, Locations: %s, Delta MSE: %f'%(iteration, ','.join(new_constant_interval_locations), delta_mse)

      else:
         optimal_constrained_mse.append(np.nan)
         opt_signal = None

      # Plot the progress
      do_plot = do_compute_mse
      if do_plot:
         plt.clf()

         if strategy == 'gradient_threshold':
            plt.subplot(321)
            (num_convex, num_total_intervals) = zip(*num_convex_intra_intervals)
            plt.bar(np.arange(len(num_convex)), num_convex)
            plt.title('Num convex vs. total intra-intervals')
         else:
            plt.subplot(321)
            iter_list = range(iteration)
            plt.plot(iter_list, iter_list)
            plt.title('Iteration %d'%(iteration))

         plt.subplot(322)
         plt.plot(constant_intervals_per_iter)
         plt.title('Number constant intervals per iteration')

         plt.subplot(323)
         plt.plot(total_interval_length_per_iter)
         plt.title('Total constant intervals length per iteration')

         plt.subplot(324)
         plt.hist(interval_lengths, bins=30)
         plt.title('Distribution of interval lengths')

         plt.subplot(325)
         plt.plot(optimal_constrained_mse)
         plt.title('Optimal Constrained MSE')

         plt.subplot(326)
         plt.plot(time, opt_signal)
         plt.plot(time, original_signal, 'r-')
         if 'new_constant_points' in locals():
            plt.plot(time[new_constant_points], opt_signal[new_constant_points], 'o')
         plt.title('Optimal Constrained Signal')

         if require_plot_input:
            plt.show()
         else:
            plt.draw()
            plt.pause(0.001)
      else:
         print('Iteration: %d'%(iteration))

      fit_signal_dict[iteration] = {'iteration_signal': signal.copy(), 'opt_fit_signal': opt_signal, 'num_constant_intervals': len(intervals)}

      if len(intervals) == 0:
         is_finished = True

      iteration = iteration + 1


   # Find and plot the elbow joints in the optimal constrained MSE plot
   #elbow_indices = GetElbowIndices(optimal_constrained_mse, 3)
   #plt.subplot(326)
   #plt.plot(elbow_indices, np.array(optimal_constrained_mse)[elbow_indices], 'ro')
   #plt.draw()
   #plt.pause(0.001)
   #for elbow_index in elbow_indices:
   #   iter_signal = fit_signal_dict[elbow_index]['iteration_signal']
   #   fit_signal = fit_signal_dict[elbow_index]['opt_fit_signal']

   #   plt.figure()
   #   plt.plot(time, fit_signal, 'b')
   #   plt.plot(time, original_signal, 'r-')
   #   plt.plot(time, iter_signal, 'g--')
   #   plt.legend(['Optimal', 'Original', 'Iteration'])
   #   plt.title('Optimal constrained linear fit: iteration %d'%(elbow_index))
   #   plt.draw()
   #   plt.pause(0.001)

   pdb.set_trace()

   # Save fit signals each iteration
   with open(out_pickle_file, 'wb') as outfile:
      pickle.dump(fit_signal_dict, outfile, pickle.HIGHEST_PROTOCOL)

   return

if __name__ == '__main__':
   if len(sys.argv) > 2:
      signal_csv_path = sys.argv[1]
      out_pickle_file = sys.argv[2]
      statprof.start()
      try:
         DoPiecewiseLinearFitting(signal_csv_path, out_pickle_file)
      finally:
         statprof.stop()
         statprof.display()
   else:
      print("Please provide the following command line arguments:\n 1) Path to signal csv file\n 2) Out pickle file")
