import os
import numpy as np
import pandas as pd
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.misc.io.base import SampleAttributes
import nibabel as nib
from collections import namedtuple
from tqdm import tqdm

OUTPUT_DIR = os.path.join("data", "derivatives", "00_roi_extraction")
# Output will be organized by mask and subject under this root directory

Condition = namedtuple("Condition", ["phase", "exp", "mask", "subj"])

subj_df = pd.read_csv("participants.tsv", sep='\t')
phases = ["A", "B"]
experiments = ["hab", "task"]
masknames = ["whole_bin"]
Conditions = [Condition(p, e, m, s)
              for p in phases
              for e in experiments
              for m in masknames
              for s in subj_df["participant_id"]]


for c in tqdm(Conditions):
	attrfile = f"{c.subj:}_phase-{c.phase:}_exp-{c.exp:}_attr.txt"
	attrpath = os.path.join("data", "raw", "pymvpa", f"{c.subj:}", attrfile)
	attr = SampleAttributes(attrpath, header=["targets", "chunks", "trial", "target_extra"])

	maskfile = f"mask-{c.mask:s}.nii.gz"
	maskpath = os.path.join("data", "raw", "roi", maskfile)

	epifile = f"{c.subj:}_phase-{c.phase:}_exp-{c.exp:}_epiMNI.nii.gz"
	epipath = os.path.join("data", "raw", "pymvpa", f"{c.subj:}", epifile)
	dataset = fmri_dataset(epipath, targets=attr.targets, chunks=attr.chunks, mask=maskpath)

	dataset.sa["trial"] = attr.trial
	dataset.sa["target_extra"] = attr.target_extra

	outdir = os.path.join(OUTPUT_DIR, f"roi-{c.mask:}", f"{c.subj:}")
	if not os.path.isdir(outdir):
		os.makedirs(outdir)

	outfile = f"{c.subj:}_phase-{c.phase:}_exp-{c.exp:}_roi-{c.mask:}.npz"
	dataset.to_npz(os.path.join(outdir, outfile))

