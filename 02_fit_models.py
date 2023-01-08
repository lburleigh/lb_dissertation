import pandas as pd
import os.path
from lb_dissertation.utils import load_npz_as_df, filter_matrix_by_set
from lb_dissertation.modeling import cv_coirls, cv_ridgels, Config


phase = "B"
experiment = "task"
roi = "whole_bin"
target_levels = ("csp", "csm") 
targets_label = "_".join(target_levels)
subj_df = pd.read_csv("participants.tsv", sep='\t')
subjects = subj_df["participant_id"]

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

d = d.drop(["trial_types_tmp", "stimulus_cond_tmp", "runs_tmp", "voxels_tmp"], axis = 1)

cfg = Config(target_field="stimulus_cond_subset", target_levels=target_levels, data_field="voxels_subset", runs_field="runs_subset")
r = []
r.append(cv_ridgels(d, True, cfg))
r[0].loc[:, "model_type"] = "ridgels"
r[0].loc[:, "cfg"] = [cfg]*r[0].shape[0]

r.append(cv_coirls(d, False, cfg))
r[1].loc[:, "model_type"] = "coirls"
r[1].loc[:, "cfg"] = [cfg]*r[1].shape[0]

r.append(cv_ridgels(d, False, cfg))
r[2].loc[:, "model_type"] = "ridgels"
r[2].loc[:, "cfg"] = [cfg]*r[2].shape[0]

R = pd.concat(r)
R.drop(["model", "cfg"], axis=1).to_csv(
    os.path.join("results", f"phase-{phase:s}_exp-{experiment:s}_roi-{roi:s}_dv-{targets_label:s}.csv")
)
R.to_pickle(
    os.path.join("results", f"phase-{phase:s}_exp-{experiment:s}_roi-{roi:s}_dv-{targets_label:s}.pkl")
)
print(R)