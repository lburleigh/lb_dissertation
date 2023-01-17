import numpy as np
from lb_dissertation.utils import load_npz_as_df
import pandas as pd
import os.path

phase = "B"
experiment = "task"
roi = "whole_bin"
target_levels = ("csp", "csm") 
targets_label = "_".join(target_levels)
subj_df = pd.read_csv("participants.tsv", sep='\t')
subjects = subj_df["participant_id"]

d = load_npz_as_df(subjects, roi, phase, experiment)

for i in d['voxels']:
    for j in range(0,31):
        #print(d.loc[j]['subject'])
        if (len(d['voxels'][j]) != 72): print("trials")
        zerovals = np.all(d['voxels'][j], axis = 0)
        if (zerovals == True).all(): print(d.loc[j]['subject'] + "_" + str(j) + "_"  + "zero")
        for k in range(0, 72):
            fvals = len(d['voxels'][j][k])
            fn = np.isnan(d['voxels'][j][k])
            if (fn == True).any(): print(d.loc[j]['subject'] + "_"  + str(j) + "_"  + "null")
            if (fvals != 193130): print(d.loc[j]['subject'] + "_"  + str(j) + "_"  + "length voxs")
