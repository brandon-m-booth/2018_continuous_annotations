#!/usr/bin/env python

import os
import sys
import pdb
import numpy as np
import pandas as pd
import statprof
import pickle
from itertools import combinations
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
      
def DoContinuousSegmentedRegression(signal_csv_path):
   signal_df = pd.read_csv(signal_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   original_signal = signal.copy()

   num_segments = 10

   knots_list = combinations(range(len(signal)), num_segments)
   print(len(list(knots_list)))

   # Save fit signals each iteration
   #with open(out_pickle_file, 'wb') as outfile:
   #   pickle.dump(fit_signal_dict, outfile, pickle.HIGHEST_PROTOCOL)

   return

if __name__ == '__main__':
   if len(sys.argv) > 1:
      signal_csv_path = sys.argv[1]
      statprof.start()
      try:
         DoContinuousSegmentedRegression(signal_csv_path)
      finally:
         statprof.stop()
         statprof.display()
   else:
      print("Please provide the following command line arguments:\n 1) Path to signal csv file")
