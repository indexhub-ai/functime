from typing import Callable, Optional, Union

import numpy as np
import polars as pl
import pyarrow as pa
from xgboost import DMatrix
from xgboost import train as xgb_train

from functime.base import forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.forecasting._ar import fit_autoreg, predict_autoreg
from functime.forecasting._regressors import GradientBoostedTreeRegressor


def _enforce_label_constraint(y: pl.DataFrame, objective: Union[str, None]):
    target_col = y.columns[-1]
    if objective == "reg:gamma":
        y = y.with_columns(
            pl.when(pl.col(target_col) <= 0)
            .then(1)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )
    elif objective in ["reg:tweedie", "count:poisson"]:
        # Fill values less than 0 with 0
        y = y.with_columns(
            pl.when(pl.col(target_col) < 0)
            .then(0)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )
    return y


def _xgboost(weight_transform: Optional[Callable] = None, **kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):

        feature_cols = X.columns[2:]

        def train(X: pa.Table, y: pa.Table, sample_weight: Optional[np.ndarray] = None):
            dataset = DMatrix(
                data=X,
                label=y,
                weight=sample_weight,
                feature_names=feature_cols,
                nthread=-1,
            )
            return xgb_train(params=kwargs, dtrain=dataset)

        regressor = GradientBoostedTreeRegressor(
            regress=train,
            weight_transform=weight_transform,
            predict_dtype=lambda X: DMatrix(
                X.select(pl.col(X.columns[2:])), feature_names=feature_cols
            ),
        )
        return regressor.fit(X=X, y=y)

    return regress


@forecaster
def xgboost(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    weight_transform: Optional[Callable] = None,
    **kwargs
):
    """Autoregressive XGBoost forecaster."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        y_new = y.pipe(_enforce_label_constraint, objective=kwargs.get("objective"))
        regress = _xgboost(weight_transform=weight_transform, **kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y_new,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg
