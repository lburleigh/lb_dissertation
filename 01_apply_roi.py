import numpy as np
import nibabel as nib
from mvpa2.suite import *
import glob

# Import FCTM data
# filelist = glob.glob("MVPAfiles/17271/*hab-epiMNI.nii.gz")

domainadaptdir = '/data/shared/TranLearn/fMRI-Domain-Adaptation'
basedir = '/data/shared/TranLearn/fMRI-Domain-Adaptation/MVPAfiles'
phase = 'A'
experiment = 'hab'
maskname = 'mask-amyg_bi_thr50.nii.gz'
subdirs=sorted(glob.glob("%s/1[0-9][0-9][0-9][0-9]"%(basedir)))

for subdir in subdirs:
 #   img_file = filename
	sub=subdir.split('/')[-1][:]
    
	## Load the sample attributes
	attrfiles=glob.glob("%s/%s-%s%s-attr.txt"%(subdir,sub,phase,experiment))[0]
	attr = SampleAttributes(attrfiles,header=['targets','chunks','trial', 'target_extra'])

    ## Find Epi and Mask files
	MNIdata = glob.glob("%s/%s-%s%s-epiMNI.nii.gz"%(subdir,sub,phase,experiment))[0]
	mask = basedir + "/" + maskname
	
    ## Create a dataeset takes the 4D dataset and transforms it into 2D,
	datset = fmri_dataset(MNIdata,targets=attr.targets,chunks=attr.chunks,mask=mask)

	## Add the additional attribute of volume
	datset.sa['trial']=attr.trial
	datset.sa['target_extra']=attr.target_extra

    ## Save file to npz
	datset.to_npz("%s/datafiles/%s/%s-%s%s.npz"%(domainadaptdir,sub,sub,phase,experiment))
