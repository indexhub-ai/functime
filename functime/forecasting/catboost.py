from typing import Callable, Optional, Union

import numpy as np
import polars as pl
from catboost import Pool
from catboost import train as cat_train

from functime.base import forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.forecasting._ar import fit_autoreg, predict_autoreg
from functime.forecasting._regressors import GradientBoostedTreeRegressor


def _enforce_label_constraint(y: pl.DataFrame, loss_function: Union[str, None]):
    target_col = y.columns[-1]
    if loss_function in ["Tweedie", "Poisson"]:
        # Fill values less than 0 with 0
        y = y.with_columns(
            pl.when(pl.col(target_col) < 0)
            .then(0)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )
    return y


def _catboost(weight_transform: Optional[Callable] = None, **kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):

        idx_cols = X.columns[:2]
        feature_cols = X.columns[2:]
        categorical_cols = X.select(pl.col(pl.Categorical).exclude(idx_cols)).columns

        def train(
            X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
        ):
            pool = Pool(
                data=X,
                label=y,
                weight=sample_weight,
                feature_names=feature_cols,
                cat_features=categorical_cols,
            )
            return cat_train(params=kwargs, pool=pool)

        regressor = GradientBoostedTreeRegressor(
            regress=train, weight_transform=weight_transform
        )
        return regressor.fit(X=X, y=y)

    return regress


@forecaster
def catboost(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    weight_transform: Optional[Callable] = None,
    **kwargs
):
    """Autoregressive Catboost forecaster."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        y_new = y.pipe(
            _enforce_label_constraint, loss_function=kwargs.get("loss_function")
        )
        regress = _catboost(weight_transform=weight_transform, **kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y_new,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg
