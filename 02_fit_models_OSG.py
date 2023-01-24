import argparse
import json
import os
import warnings
import pandas as pd
from lb_dissertation.utils import load_npz_as_df, filter_matrix_by_set, allzeros_across_all_runs, flatten_list 
from lb_dissertation.modeling import cv_coirls, cv_ridgels, DataCfg, HyperCfg

parser = argparse.ArgumentParser(description='Fit models for LB dissertation')
parser.add_argument('--roi', type=str, default="whole_bin", help='Which ROI?')
parser.add_argument('--lambda', dest = "lambda_", type=float, help='Hyperparameter to scale the CoIR penalty')
parser.add_argument('--alpha', type=float, help='Hyperparameter to scale the ridge penalty')
parser.add_argument('--phase', type=str, default="B", help='Either "A" or "B"')
parser.add_argument('--trial_types', type=str, default=["image", "view"], nargs='+', help='"image" or "view" or both')
parser.add_argument('--model_type', type=str, default="coirls", help='"coirls" or "ridge"')
parser.add_argument('--single', action='store_true', help='Should the model be fit to a single subject?')
parser.add_argument('--pickle', action='store_true', help='Should the full dataframe be pickled? This is the only way to access the model weights.')
parser.add_argument('--job_id', type=int, help='Unique job index')
parser.add_argument('--exclude_fold', type=int, default=-1, help='A fold to exclude while doing nested cross validation')
parser.add_argument('--hypfile', type=str, help='A json txt file specifying hyperparameters for each subject and cv')
args = parser.parse_args()


if ((args.lambda_ is not None) or (args.alpha is not None)) and (args.hypfile is not None):
    raise Exception("Hyperparameters cannot be specified on the command line and in a json simultaneously.")


experiment = "task"
roi = args.roi
target_levels = ("csp", "csm") 
targets_label = "_".join(target_levels)
subj_df = pd.read_csv("participants.tsv", sep='\t')
subjects = subj_df["participant_id"]

phase = args.phase
trial_types = args.trial_types
alpha = args.alpha
model_type = args.model_type
single = args.single
job_id = args.job_id
exclude_fold = args.exclude_fold

if args.lambda_ is not None and model_type == "ridge":
    warnings.warn("A value for lambda was provided, but ridge will ignore it.", RuntimeWarning)
    lambda_ = 0

elif args.lambda_ is None:
    lambda_ = 0

else:
    lambda_ = args.lambda_


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
        set = trial_types,
        axis = 1
    )

z = allzeros_across_all_runs(d)
d.voxels_subset = [x[:,~z] for x in d.voxels_subset]

d = d.drop(["trial_types_tmp", "stimulus_cond_tmp", "runs_tmp", "voxels_tmp"], axis = 1)

cfg = DataCfg(target_field="stimulus_cond_subset", target_levels=target_levels, data_field="voxels_subset", runs_field="runs_subset", exclude_fold=exclude_fold)

if args.hypfile is None:
    outdir = "tune"
    hyp = HyperCfg(alpha=alpha, lambda_=lambda_)

    if model_type == "coirls":
        r = cv_coirls(d, False, cfg, hyp)
    elif model_type == "ridge":
        r = cv_ridgels(d, single, cfg, hyp)

    r.loc[:, "model_type"] = model_type
    r.loc[:, "cfg"] = [cfg]*r.shape[0]
    r.loc[:, "hyp"] = [hyp]*r.shape[0]

else:
    outdir = "final"
    with open(args.hypfile, 'r') as f:
        hyplist = json.load(f)

    if isinstance(hyplist, dict):
        hyplist = [[HyperCfg(alpha=k['alpha'], lambda_=k['lambda']) for k in hyplist[s]] for s in d.subject]

    if model_type == "coirls":
        r = cv_coirls(d, False, cfg, hyplist)
    elif model_type == "ridge":
        r = cv_ridgels(d, single, cfg, hyplist)

    r.loc[:, "model_type"] = model_type
    r.loc[:, "cfg"] = [cfg]*r.shape[0]
    r.loc[:, "hyp"] = flatten_list(hyplist)

os.makedirs(os.path.join("results", outdir), exist_ok=True)
r.drop(["model_params", "model_weights", "cfg"], axis=1).to_csv(
    os.path.join(
        "results",
        outdir,
        f"phase-{phase:s}_exp-{experiment:s}_roi-{roi:s}_dv-{targets_label:s}_model-{model_type:s}_single-{single:d}_exclude-{exclude_fold:d}_job-{job_id:d}.csv")
)

if args.pickle:
    r.to_pickle(
        os.path.join(
            "results",
            outdir,
            f"phase-{phase:s}_exp-{experiment:s}_roi-{roi:s}_dv-{targets_label:s}_model-{model_type:s}_single-{single:d}_exclude-{exclude_fold:d}_job-{job_id:d}.pkl")
    )
