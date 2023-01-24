import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from collections import namedtuple
from typing import Callable, Union, List
from sklearn.metrics import accuracy_score, log_loss
from kale.pipeline.multi_domain_adapter import CoIRLS


DataCfg = namedtuple("DataCfg", ["target_field", "target_levels", "data_field", "runs_field", "exclude_fold"])
Data = namedtuple("Data", ["X", "y", "C", "cv", "source", "target_subject", "exclude_fold"])
HyperCfg = namedtuple("HyperCfg", ["alpha", "lambda_"])
Result = namedtuple("Result", ["target_subject", "cv_index", "exclude_fold", "single", "acc_test", "acc_train", "loss_test", "loss_train", "model_params", "model_weights"])


def cv_modelfit(fun: Callable[[Data, int], Result], d: pd.DataFrame, single: bool, cfg: DataCfg, hyp: Union[HyperCfg, List[List[HyperCfg]]]) -> pd.DataFrame:
    results = []
    for target_subject_index in trange(d.shape[0], desc="subject"):
        if single:
            data = pull_from_dataframe(d.iloc[[target_subject_index], :], 0, cfg)
        else:
            data = pull_from_dataframe(d, target_subject_index, cfg)

        cv_set = np.unique(data.cv[~data.source])
        for cv_index in tqdm(cv_set, desc="cv", leave=False):
            if isinstance(hyp, HyperCfg):
                results.append(fun(data, cv_index, single, hyp))
            else:
                results.append(fun(data, cv_index, single, hyp[target_subject_index][cv_index]))

    return pd.DataFrame(results)


def cv_coirls(d: pd.DataFrame, single: bool, cfg: DataCfg, hyp: HyperCfg) -> pd.DataFrame:
    return cv_modelfit(run_coirls, d, single, cfg, hyp)
    

def cv_ridgels(d: pd.DataFrame, single: bool, cfg: DataCfg, hyp: HyperCfg) -> pd.DataFrame:
    return cv_modelfit(run_ridgels, d, single, cfg, hyp)
    

def pull_from_dataframe(d: pd.DataFrame, target_subject_index: int, cfg: DataCfg) -> Data:
    y_str = np.concatenate(d[cfg.target_field].values, axis=0)
    y = (y_str == cfg.target_levels[1]).astype(int)
    cv = np.concatenate(d[cfg.runs_field].values, axis=0).flatten()
    X = np.concatenate(d[cfg.data_field].values, axis=0)
    sub_index = np.concatenate(
        [[i]*x.shape[0] for i,x in enumerate(d[cfg.data_field])],
        axis=0
    )
    C = np.identity(d.shape[0])[sub_index,:]

    # Modify based on target subject (make sure target occupies bottom rows)
    target_subject = d.subject.iloc[target_subject_index]
    min_cv = np.min(cv)
    z_source = sub_index != target_subject_index
    z_exclude = ((cv == cfg.exclude_fold) & ~z_source)

    X = X[~z_exclude, :]
    C = C[~z_exclude, :]
    y = y[~z_exclude]
    z_source = z_source[~z_exclude]
    cv = cv[~z_exclude]
    return Data(X, y, C, cv-min_cv, z_source, target_subject, cfg.exclude_fold-min_cv)


def run_coirls(data: Data, cv_index: int, single: bool, hyp: HyperCfg) -> Result:
    clf_ = CoIRLS(alpha=hyp.alpha, lambda_=hyp.lambda_)
    z_train = data.source | (data.cv != cv_index)
    X = (data.X - data.X[z_train, :].mean(axis=0)) / data.X[z_train, :].std(axis=0)
    X_test = X[~z_train, :]
    X_train =X[z_train, :]
    y_test = data.y[~z_train]
    y_train = data.y[z_train]
    X = np.concatenate([X_train, X_test], axis=0)

    clf_.fit(X, y_train, data.C)
    y_train_pred = clf_.predict(X_train)
    y_test_pred = clf_.predict(X_test)
    return Result(
        data.target_subject,
        cv_index,
        data.exclude_fold,
        single,
        accuracy_score(y_test, y_test_pred),
        accuracy_score(y_train, y_train_pred),
        log_loss(y_test, y_test_pred),
        log_loss(y_train, y_train_pred),
        clf_.get_params(),
        clf_.coef_.numpy()
    )


def run_ridgels(data: Data, cv_index: int, single: bool, hyp: HyperCfg) -> Result:
    clf_ = CoIRLS(alpha=hyp.alpha, lambda_=hyp.lambda_)
    z_train = data.source | (data.cv != cv_index)
    X = (data.X - data.X[z_train, :].mean(axis=0)) / data.X[z_train, :].std(axis=0)
    X_test = X[~z_train, :]
    X_train =X[z_train, :]
    y_test = data.y[~z_train]
    y_train = data.y[z_train]
    C = np.ones([len(y_train), 1])

    clf_.fit(X_train, y_train, C)
    y_train_pred = clf_.predict(X_train)
    y_test_pred = clf_.predict(X_test)
    return Result(
        data.target_subject,
        cv_index,
        data.exclude_fold,
        single,
        accuracy_score(y_test, y_test_pred),
        accuracy_score(y_train, y_train_pred),
        log_loss(y_test, y_test_pred),
        log_loss(y_train, y_train_pred),
        clf_.get_params(),
        clf_.coef_.numpy()
    )