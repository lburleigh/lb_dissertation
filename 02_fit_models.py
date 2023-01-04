import pandas as pd
from lb_dissertation.utils import load_npz_as_df, filter_matrix_by_set
from lb_dissertation.modeling import cv_coirls, Config


phase = "B"
experiment = "task"
roi = "amyg_bi_thr50"
target_levels = ("csp", "csm") 
subj_df = pd.read_csv("participants.tsv", sep='\t')
subjects = subj_df["participant_id"]

d = load_npz_as_df(subjects, roi, phase, experiment)

# Filter arrays within each row
for var in ["trial_types", "stimulus_cond", "runs", "voxels"]:
    d[f"{var:s}_subset"] = d.apply(
        filter_matrix_by_set,
        var = var,
        by = "stimulus_cond",
        set = target_levels,
        axis = 1
    )

cfg = Config(target_field="stimulus_cond_subset", target_levels=target_levels, data_field="voxels_subset", runs_field="runs")
r = cv_coirls(d, cfg)
print(r)