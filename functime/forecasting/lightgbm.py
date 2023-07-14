from typing import Callable, Optional, Union

import numpy as np
import polars as pl
from lightgbm import Dataset
from lightgbm import train as lgb_train

from functime.base import forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.forecasting._ar import fit_autoreg, predict_autoreg
from functime.forecasting._regressors import GradientBoostedTreeRegressor


def _prepare_kwargs(kwargs):
    new_kwargs = {}
    tree_learner = kwargs.get("tree_learner") or "serial"
    new_kwargs["tree_learner"] = tree_learner
    new_kwargs["verbose"] = -1
    new_kwargs["force_col_wise"] = True
    alpha = new_kwargs.get("alpha")
    if alpha is not None:
        new_kwargs["objective"] = "quantile"
    return {**new_kwargs, **kwargs}


def _enforce_label_constraint(y: pl.DataFrame, objective: Union[str, None]):
    target_col = y.columns[-1]
    if objective == "gamma":
        y = y.with_columns(
            pl.when(pl.col(target_col) <= 0)
            .then(1)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )
    elif objective in ["tweedie", "poisson"]:
        # Fill values less than 0 with 0
        y = y.with_columns(
            pl.when(pl.col(target_col) < 0)
            .then(0)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )
    return y


def _lightgbm(weight_transform: Optional[Callable] = None, **kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):

        idx_cols = X.columns[:2]
        feature_cols = X.columns[2:]
        categorical_cols = X.select(pl.col(pl.Categorical).exclude(idx_cols)).columns

        def train(
            X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
        ):
            dataset = Dataset(
                data=X,
                label=y,
                weight=sample_weight,
                feature_name=feature_cols,
                categorical_feature=categorical_cols,
            )
            return lgb_train(params=params, train_set=dataset)

        params = _prepare_kwargs(kwargs)
        regressor = GradientBoostedTreeRegressor(
            regress=train, weight_transform=weight_transform
        )
        return regressor.fit(X=X, y=y)

    return regress


def _flaml_lightgbm(
    time_budget: Optional[int] = None,
    max_iter: Optional[int] = None,
    weight_transform: Optional[Callable] = None,
    **kwargs
):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from flaml import AutoML

        tuner_kwargs = {
            "time_budget": time_budget,
            "max_iter": max_iter,
            "metric": "rmse",
            "estimator_list": ["lgbm"],
            "task": "regression",
            "split_type": "time",
        }
        regressor_kwargs = _prepare_kwargs(kwargs)
        tuner = AutoML(
            **tuner_kwargs,
            custom_hp={
                "lgbm": {
                    param: {"domain": value}
                    for param, value in regressor_kwargs.items()
                }
            },
        )
        sample_weights = None
        if weight_transform is not None:
            sample_weights = y.pipe(weight_transform)
        tuner.fit(
            X_train=X.to_pandas(), y_train=y.to_pandas(), sample_weight=sample_weights
        )
        return tuner

    return regress


@forecaster
def lightgbm(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    weight_transform: Optional[Callable] = None,
    **kwargs
):
    """Autoregressive LightGBM forecaster."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        y_new = y.pipe(_enforce_label_constraint, objective=kwargs.get("objective"))
        regress = _lightgbm(weight_transform=weight_transform, **kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y_new,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg


@forecaster
def flaml_lightgbm(
    freq: Union[str, None],
    lags: int,
    time_budget: Optional[int] = None,
    max_iter: Optional[int] = None,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    weight_transform: Optional[Callable] = None,
    **kwargs
):
    """Autoregressive FLAML AutoML LightGBM forecaster."""

    time_budget = time_budget or 30
    max_iter = max_iter or 30

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        y_new = y.pipe(_enforce_label_constraint, objective=kwargs.get("objective"))
        regress = _flaml_lightgbm(
            weight_transform=weight_transform,
            time_budget=time_budget,
            max_iter=max_iter,
            **kwargs,
        )
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y_new,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg
