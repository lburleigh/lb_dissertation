# Script to merge csvs

import os
import pandas as pd
import glob

path = '/data/shared/TranLearn/lb_dissertation/results' # use your path

csv_files = glob.glob('*.{}'.format('csv'))
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df.to_csv('/data/shared/TranLearn/lb_dissertation/results/all_hyperparameters.csv') 
