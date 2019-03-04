#!/usr/bin/env python

import os
import sys
import pwlf
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import statprof

num_segments = 38

def DoPWLF(input_csv_path):
   # Get the signal data
   signal_df = pd.read_csv(input_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]

   # Fit a PWLF to the data with num_segments
   pwlf_obj = pwlf.PiecewiseLinFit(time, signal)
   print("Beginning profile of pwlf fit:")
   statprof.start()
   try:
      seg_reg = pwlf_obj.fit(num_segments)
   finally:
      statprof.stop()
      statprof.display()
   print("Finished timing pwlf fit")

   # Get vector of fitted signal values
   signal_fit = pwlf_obj.predict(time)

   # Plot results
   plt.figure()
   plt.plot(time, signal, 'b-')
   plt.plot(time, signal_fit, 'r--')
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
   DoPWLF(input_csv_path)
