## Packages/libraries
import numpy as np
import glob
import pandas as pd
import os

## Denote directories and such
domainadaptdir = '/data/shared/TranLearn/fMRI-Domain-Adaptation'
basedir = '/data/shared/TranLearn/fMRI-Domain-Adaptation/datafiles'
phase = 'B'
experiment = 'task'
subdirs=sorted(glob.glob("%s/1[0-9][0-9][0-9][0-9]"%(basedir)))

## Create empty arrays to append to
#dfvoxs = np.empty([48,520])
#dfsubs = np.zeros([48,520])

d = pd.DataFrame(
	{
     "subject": [os.path.basename(subdir) for subdir in subdirs],
     "filepath": [
            os.path.join("datafiles", subj, f"{subj:s}-{phase:s}{experiment:s}.npz")
            for subj in [os.path.basename(subdir) for subdir in subdirs]
        ]
    }
)

tmp = [np.load(f) for f in d.filepath]
# samples (voxels), sa.targets (+,-), sa.target_extra (trial type), sa.chunks (run)
# All 1-D arrays are forced to be column vectors
d["trial_types"] = [x["sa.target_extra"][:, np.newaxis] for x in tmp]
d["stimulus_cond"] = [x["sa.targets"][:, np.newaxis] for x in tmp]
d["runs"] = [x["sa.chunks"][:, np.newaxis] for x in tmp]
d["voxels"] = [x["samples"] for x in tmp]
[x.close() for x in tmp]
del tmp


# Define simple subsetting functions to be applied to arrays within each row of
# the data frame.
def subset_array_by_val(row, var, by, val):
    return row[var][[x == val for x in row[by]], :]


def subset_array_by_set(row, var, by, set):
    return row[var][[x in set for x in row[by]], :]


# Filter arrays within each row
cond_set = (b"csp", b"csm") 
for var in ["trial_types", "stimulus_cond", "runs", "voxels"]:
    d[f"{var:s}_subset"] = d.apply(
        subset_array_by_set,
        var = var,
        by = "stimulus_cond",
        set = cond_set,
        axis = 1
    )


# Extract the variables necessary for modeling
y = np.concatenate(d["trial_types_subset"].values, axis = 0)
X = np.concatenate(d["voxels_subset"].values, axis = 0)
sub_index = np.concatenate(
    [[i]*x.shape[0] for i,x in enumerate(d["voxels_subset"])],
    axis = 0
)
C = np.identity(d.shape[0])[sub_index,:]
target_subject = 0
D = np.array([1 if i == 0 else 0 for i in sub_index])


# Begin modeling ....


for subdir in subdirs:
	sub=subdir.split('/')[-1][:]

    ## Load npz and data of interest
	datafile=np.load("%s/%s/%s-%s%s.npz"%(basedir,sub,sub,phase,experiment))
	voxels=datafile['samples']
    # samples (voxels), sa.targets (+,-), sa.target_extra (trial type), sa.chunks (run)
	

	
	## appending numpys
	#dfvoxs = np.append(dfvoxs, voxels, axis=0)
	#dfsubs = np.append(dfsubs, xx, axis=1) #source v target
 




 ######## Spare Meeting Notes ########
 # third matrix of p v m
 # pandas dataframe - as loading, add cell with row info ? 
	# have list and within list have multiple dictionaries. 

# coir(data, subs, covar: + v -)

# domain adapt with covar, swap covar with cols of 1s [no target/source denotion], 
# only give target and no source domain data, extra: apply PCA

# ds_train.to_npz('filename') to npz
# x=numpy.load('filename.npz')
# samples [voxels], targets, target_extra, chunks
# concatenate by row all datasets that will be in analysis, 
# alongsize covariate [1,0s with 1 for indv sub for each col]
# target domain - split to train/test, fit model on train but take into account all source domain to give model to make predictions on test
    # iterate through targets, holding source, take average. repeat for subjects. 
    # avg cross validated for each subject