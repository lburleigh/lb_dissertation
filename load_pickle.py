import pandas as pd
import pickle

with open("results/phase-B_exp-task_roi-whole_bin_dv-csp_csm.pkl", 'rb') as f:
    d = pickle.load(f)