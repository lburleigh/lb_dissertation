"""Collection of basic utility functions"""

import os
import pandas as pd
import numpy as np
from typing import Iterable

def filter_matrix_by_set(row: pd.Series, var: str, by: str, set: Iterable[str]):
    """ Apply a filter to matrices in each row of a data frame

    Parameters
    ----------
    row: pd.Series
    var: str
        The name of a variable in the data frame containing a 2-D array (matrix)
        to be filtered row-wise.
    by: str
        The name of a variable in the data frame containing a 1-D array or
        column vector of values that will be used to define the filter.
    set: Iterable[str]
        A set of values that exist in `by`.

    Returns
    -------
    np.array
    
    """
    return row[var][[x in set for x in row[by]], :]


def load_npz_as_df(subjects: Iterable, roi: str, phase: str, exp: str) -> pd.DataFrame:
    """Load npz files for a set of subjects as a data frame

    The npz files this function is intended to load are generated by PyMPVA, and
    contain several objects:
    - samples (voxels)
    - sa.targets (+,-)
    - sa.target_extra (trial type),
    - sa.chunks (run)

    Parameters
    ----------
    subjects: Iterable
        A list of participant ids.
    roi: str
        An ROI label.
    phase: str
        A phase label.
    exp: str
        An experiment label.

    Returns
    -------
    pd.DataFrame
        All 1-D arrays are forced to be column vectors

    """
    d = pd.DataFrame(
        {
        "subject": subjects,
        "filepath": [
                os.path.join("data", "derivatives", "00_roi_extraction", f"roi-{roi:s}", subj, f"{subj:s}_phase-{phase:s}_exp-{exp:s}_roi-{roi:s}.npz")
                for subj in subjects 
            ]
        }
    )

    NpzFiles = [np.load(f) for f in d.filepath]
    d["trial_types"] = [x["sa.target_extra"][:, np.newaxis] for x in NpzFiles]
    d["stimulus_cond"] = [x["sa.targets"][:, np.newaxis] for x in NpzFiles]
    d["runs"] = [x["sa.chunks"][:, np.newaxis] for x in NpzFiles]
    d["voxels"] = [x["samples"] for x in NpzFiles]
    [x.close() for x in NpzFiles]

    return d


def allzeros_across_all_runs(d: pd.DataFrame):
    runs = np.unique(d.runs_subset[0])
    z = []
    for i in runs:
        z.extend([np.all(y[x.flatten() == i, :] == 0, axis=0) for x,y in zip(d.runs_subset, d.voxels_subset)])
    
    return np.any(np.array(z), axis=0).flatten()
        