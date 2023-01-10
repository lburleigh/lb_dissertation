import numpy as np
import pandas as pd
from tqdm import trange
from collections import namedtuple
from typing import Callable
from sklearn.metrics import accuracy_score, log_loss
from kale.pipeline.multi_domain_adapter import CoIRLS


Config = namedtuple("Config", ["target_field", "target_levels", "data_field", "runs_field"])
Data = namedtuple("Data", ["X", "y", "C", "cv", "source", "target_subject"])
Result = namedtuple("Result", ["target_subject", "cv_index", "single", "acc_test", "acc_train", "loss_test", "loss_train", "model_params", "model_weights"])


def cv_modelfit(fun: Callable[[Data, int], Result], d: pd.DataFrame, single: bool, cfg: Config) -> pd.DataFrame:
    results = []
    for target_subject_index in trange(d.shape[0], desc="subject"):
        if single:
            data = pull_from_dataframe(d.iloc[[target_subject_index], :], 0, cfg)
        else:
            data = pull_from_dataframe(d, target_subject_index, cfg)

        for cv_index in trange(np.max(data.cv), desc="cv", leave=False):
            results.append(fun(data, cv_index, single))

    return pd.DataFrame(results)


def cv_coirls(d: pd.DataFrame, single: bool, cfg: Config) -> pd.DataFrame:
    return cv_modelfit(run_coirls, d, single, cfg)
    

def cv_ridgels(d: pd.DataFrame, single: bool, cfg: Config) -> pd.DataFrame:
    return cv_modelfit(run_ridgels, d, single, cfg)
    

def pull_from_dataframe(d: pd.DataFrame, target_subject_index: int, cfg: Config) -> Data:
    y_str = np.concatenate(d[cfg.target_field].values, axis = 0)
    y = (y_str == cfg.target_levels[1]).astype(int)
    cv = np.concatenate(d[cfg.runs_field].values, axis = 0).flatten()
    X = np.concatenate(d[cfg.data_field].values, axis = 0)
    sub_index = np.concatenate(
        [[i]*x.shape[0] for i,x in enumerate(d[cfg.data_field])],
        axis = 0
    )
    C = np.identity(d.shape[0])[sub_index,:]

    # Modify based on target subject (make sure target occupies bottom rows)
    target_subject = d.subject.iloc[target_subject_index]
    z_source = np.array([False if i == target_subject_index else True for i in sub_index])
    X = np.concatenate([X[z_source, :], X[~z_source, :]], axis = 0)
    y = np.concatenate([y[z_source], y[~z_source]], axis = 0)
    C = np.concatenate([C[z_source, :], C[~z_source, :]])
    cv = np.concatenate([cv[z_source], cv[~z_source]]) - np.min(cv) # ensure zero based
    z_source = np.concatenate([z_source[z_source], z_source[~z_source]])

    return Data(X, y, C, cv, z_source, target_subject)


def run_coirls(data: Data, cv_index: int, single: bool) -> Result:
    clf_ = CoIRLS()
    z_train = data.source | (data.cv != cv_index)
    X_test = data.X[~z_train, :]
    X_train = data.X[z_train, :]
    y_test = data.y[~z_train]
    y_train = data.y[z_train]

    clf_.fit(data.X, y_train, data.C)
    y_train_pred = clf_.predict(X_train)
    y_test_pred = clf_.predict(X_test)
    return Result(
        data.target_subject,
        cv_index,
        single,
        accuracy_score(y_test, y_test_pred),
        accuracy_score(y_train, y_train_pred),
        log_loss(y_test, y_test_pred),
        log_loss(y_train, y_train_pred),
        clf_.get_params(),
        clf_.coef_.numpy()
    )


def run_ridgels(data: Data, cv_index: int, single: bool) -> Result:
    clf_ = CoIRLS()
    z_train = data.source | (data.cv != cv_index)
    X_test = data.X[~z_train, :]
    X_train = data.X[z_train, :]
    y_test = data.y[~z_train]
    y_train = data.y[z_train]
    C = np.ones([len(y_train), 1])

    clf_.fit(X_train, y_train, C)
    y_train_pred = clf_.predict(X_train)
    y_test_pred = clf_.predict(X_test)
    return Result(
        data.target_subject,
        cv_index,
        single,
        accuracy_score(y_test, y_test_pred),
        accuracy_score(y_train, y_train_pred),
        log_loss(y_test, y_test_pred),
        log_loss(y_train, y_train_pred),
        clf_.get_params(),
        clf_.coef_.numpy()
    )