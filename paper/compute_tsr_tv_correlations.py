#!/usr/bin/env python

import re
import os
import sys
import pdb
import glob
import shutil
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics

def ComputeTSRTVCorrelations():
   # TODO - fix hard-coded paths
   tsr_folder_path = '/USC/2018_Continuous_Annotations/data/GreenIntensityTasks/tsr'
   tv_folder_path = '/USC/2018_Continuous_Annotations/data/GreenIntensityTasks/tv'
   scripts_folder = '/USC/2016_Continuous_Annotations/scripts/'
   resample_script = '/USC/2016_Engagement_Pilot/scripts/resample_csv.py'
   tste_path = '/USC/2016_Continuous_Annotations/scripts/ordinal_embedding/tste/'
   task_a_truth = '/USC/2016_Continuous_Annotations/annotation_tasks/TaskA/AnnotationData/objective_truth/TaskA_normalized_1hz.csv'
   task_b_truth = '/USC/2016_Continuous_Annotations/annotation_tasks/TaskB/AnnotationData/objective_truth/TaskB_normalized_1hz.csv'
   eval_dep_task_a = '/USC/2016_Continuous_Annotations/annotation_tasks/TaskA/AnnotationData/ground_truth_baselines/eval_dep/eval_dep_ground_truth_1hz.csv'
   eval_dep_task_b = '/USC/2016_Continuous_Annotations/annotation_tasks/TaskB/AnnotationData/ground_truth_baselines/eval_dep/eval_dep_ground_truth_1hz.csv'
   output_corr_path = '/USC/2018_Continuous_Annotations/paper/'

   for method in ['tsr', 'tv']:
      if method == 'tsr':
         data_folder_path = tsr_folder_path
      elif method == 'tv':
         data_folder_path = tv_folder_path

      # Make output folders
      resampled_path = os.path.join(data_folder_path, 'resampled_1hz')
      intervals_path = os.path.join(data_folder_path, 'intervals')
      ordinated_intervals_path = os.path.join(data_folder_path, 'ordinated_intervals')
      warped_path = os.path.join(data_folder_path, 'warped')
      if os.path.exists(resampled_path):
         shutil.rmtree(resampled_path)
      if os.path.exists(intervals_path):
         shutil.rmtree(intervals_path)
      if os.path.exists(ordinated_intervals_path):
         shutil.rmtree(ordinated_intervals_path)
      if os.path.exists(warped_path):
         shutil.rmtree(warped_path)
      os.makedirs(resampled_path)
      os.makedirs(intervals_path)
      os.makedirs(ordinated_intervals_path)
      os.makedirs(warped_path)

      # Correlation data storage
      correlations = {'TaskA': {}, 'TaskB': {}}

      # Resample to 1hz
      os.system('python '+resample_script+' '+data_folder_path+' 1 '+resampled_path+' linear')

      data_resampled_files = glob.glob(os.path.join(resampled_path, '*.csv'))
      for data_resampled_file in data_resampled_files:
         file_name = os.path.basename(data_resampled_file)
         print("Processing file: "+file_name)
         if method == 'tsr':
            file_name_re_search = re.search('opt_trapezoid_(\d+)_segments', file_name, re.IGNORECASE)
            method_param = int(file_name_re_search.group(1))
         elif method == 'tv':
            file_name_re_search = re.search('tv_(\d+.\d+)_lambda', file_name, re.IGNORECASE)
            method_param = float(file_name_re_search.group(1))

         if 'taska' in file_name.lower():
            objective_truth_path = task_a_truth
            eval_dep_path = eval_dep_task_a
            task = 'TaskA'
         elif 'taskb' in file_name.lower():
            objective_truth_path = task_b_truth
            eval_dep_path = eval_dep_task_b
            task = 'TaskB'
         else:
            print("FIX ME")
            pdb.set_trace()

         # Constant interval extraction
         intervals_file_out = os.path.join(intervals_path,file_name[:-4]+'_intervals.csv')
         strict_str = ' strict' if method == 'tsr' else ''
         os.system('python '+os.path.join(scripts_folder,'compute_constant_intervals.py')+' '+data_resampled_file+' '+intervals_file_out+strict_str)

         # Check that there are enough intervals to proceed
         intervals_df = pd.read_csv(intervals_file_out, header=None)
         if intervals_df.shape[0] < 3:
            continue
         num_segments = intervals_df.shape[0]

         # Ordinate intervals using oracle
         ordinated_file_out = os.path.join(ordinated_intervals_path, file_name[:-4]+'_ordinated.csv')
         matlab_command = 'matlab -nodisplay -r "cd(\''+tste_path+'\'); ordinateIntervals(\''+ordinated_file_out+'\', \''+data_resampled_file+'\', \''+objective_truth_path+'\', \''+intervals_file_out+'\');exit"'
         os.system(matlab_command)
         
         # Warp signal
         warped_file_out = os.path.join(warped_path, file_name[:-4]+'_warped.csv')
         os.system('python '+os.path.join(scripts_folder,'warp_signal.py')+' '+eval_dep_path+' '+intervals_file_out+' '+ordinated_file_out+' '+objective_truth_path+' '+warped_file_out)

         # Correlation measures
         warped_signal = pd.read_csv(warped_file_out)
         truth_signal = pd.read_csv(objective_truth_path)
         truncate_length = min(warped_signal.shape[0], truth_signal.shape[0])
         warped_signal = warped_signal['Data'][0:truncate_length]
         truth_signal = truth_signal['Data'][0:truncate_length]
         pearson_corr = scipy.stats.pearsonr(warped_signal, truth_signal)[0]
         spearman_corr = scipy.stats.spearmanr(truth_signal, warped_signal)[0]
         kendall_tau = scipy.stats.kendalltau(truth_signal, warped_signal)[0]
         nmi = sklearn.metrics.normalized_mutual_info_score(truth_signal, warped_signal)

         # Store the results
         correlations[task][method_param] = (pearson_corr, spearman_corr, kendall_tau, nmi, num_segments)

      # Save the results 
      for task in correlations.keys():
         output_corr_file = os.path.join(output_corr_path, method+'_'+task+'_correlations.csv')
         num_unique_params = len(correlations[task].keys())
         correlation_mat = np.zeros((num_unique_params,6))
         params = sorted(correlations[task].keys())
         for i in range(len(params)):
            param= params[i]
            correlation_mat[i,0] = param
            correlation_mat[i,1:] = correlations[task][param]
         df = pd.DataFrame(data=correlation_mat, columns=['Method Param', 'Pearson', 'Spearman', 'Kendall Tau', 'NMI', 'Num Segments'])
         df.to_csv(output_corr_file, index=False, header=True)

ComputeTSRTVCorrelations()
