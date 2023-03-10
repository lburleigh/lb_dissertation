import pandas as pd
import os.path
from itertools import product
from lb_dissertation.utils import load_npz_as_df, filter_matrix_by_set, allzeros_across_all_runs 
from lb_dissertation.modeling import cv_coirls, cv_ridgels, DataCfg, HyperCfg
from itertools import product


phase = "B"
experiment = "task"
roi = "whole_bin"
target_levels = ("csp", "csm") 
targets_label = "_".join(target_levels)
subj_df = pd.read_csv("participants.tsv", sep='\t')
subjects = subj_df["participant_id"]

alpha_set=[.001, .01, 1, 10, 100, 500, 1000]
# , .01, 1, 10, 100, 500, 1000
lambda_set=[.001, .01, 1, 10, 100, 500, 1000]

d = load_npz_as_df(subjects, roi, phase, experiment)

# Filter arrays within each row
for var in ["trial_types", "stimulus_cond", "runs", "voxels"]:
    d[f"{var:s}_tmp"] = d.apply(
        filter_matrix_by_set,
        var = var,
        by = "stimulus_cond",
        set = target_levels,
        axis = 1
    )

for var in ["trial_types_tmp", "stimulus_cond_tmp", "runs_tmp", "voxels_tmp"]:
    lab = var.replace("_tmp", "")
    d[f"{lab:s}_subset"] = d.apply(
        filter_matrix_by_set,
        var = var,
        by = "trial_types_tmp",
        set = ("image", "view"),
        axis = 1
    )

z = allzeros_across_all_runs(d)
d.voxels_subset = [x[:,~z] for x in d.voxels_subset]

d = d.drop(["trial_types_tmp", "stimulus_cond_tmp", "runs_tmp", "voxels_tmp"], axis = 1)

cfg = DataCfg(target_field="stimulus_cond_subset", target_levels=target_levels, data_field="voxels_subset", runs_field="runs_subset")
HyperCfgs =[HyperCfg(alpha=x, lambda_=y) for x,y in product(alpha_set, lambda_set)]
r = []
for hyp in HyperCfgs:
    r.append(cv_coirls(d, False, cfg, hyp))
    r[-1].loc[:, "model_type"] = "coirls"
    r[-1].loc[:, "cfg"] = [cfg]*r[-1].shape[0]
    r[-1].loc[:, "hyp"] = [hyp]*r[-1].shape[0]


#HyperCfgs =[HyperCfg(alpha=x, lambda_=y) for x,y in product(alpha_set, [0])]
#for hyp in HyperCfgs:
#    r.append(cv_ridgels(d, True, cfg, hyp))
#    r[-1].loc[:, "model_type"] = "ridgels"
#    r[-1].loc[:, "cfg"] = [cfg]*r[-1].shape[0]
#    r[-1].loc[:, "hyp"] = [hyp]*r[-1].shape[0]

#    r.append(cv_ridgels(d, False, cfg, hyp))
#    r[-1].loc[:, "model_type"] = "ridgels"
#    r[-1].loc[:, "cfg"] = [cfg]*r[-1].shape[0]
#    r[-1].loc[:, "hyp"] = [hyp]*r[-1].shape[0]


R = pd.concat(r)
R.drop(["model_params", "model_weights", "cfg"], axis=1).to_csv(
    os.path.join("results", f"phase-{phase:s}_exp-{experiment:s}_roi-{roi:s}_dv-{targets_label:s}_hyperconfigs-coir.csv")
)
R.to_pickle(
    os.path.join("results", f"phase-{phase:s}_exp-{experiment:s}_roi-{roi:s}_dv-{targets_label:s}_hyperconfigs-coir.pkl")
)
print(R)