from typing import Optional, Union

import polars as pl

from functime.base import forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.forecasting._ar import fit_autoreg, predict_autoreg
from functime.forecasting._regressors import StandardizedSklearnRegressor


def _linear_model(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import LinearRegression

        regressor = StandardizedSklearnRegressor(
            estimator=LinearRegression(**kwargs, copy_X=False),
        )
        return regressor.fit(X=X, y=y)

    return regress


def _lasso(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import Lasso

        regressor = StandardizedSklearnRegressor(
            estimator=Lasso(**kwargs, tol=0.001, copy_X=False, max_iter=10000),
        )
        return regressor.fit(X=X, y=y)

    return regress


def _ridge(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import Ridge

        regressor = StandardizedSklearnRegressor(
            estimator=Ridge(**kwargs, tol=0.001, copy_X=False, max_iter=10000)
        )
        return regressor.fit(X=X, y=y)

    return regress


def _elastic_net(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import ElasticNet

        regressor = StandardizedSklearnRegressor(
            estimator=ElasticNet(**kwargs, tol=0.001, copy_X=False, max_iter=10000)
        )
        return regressor.fit(X=X, y=y)

    return regress


@forecaster
def linear_model(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Linear autoregressive model."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):

        # Check dummy variable trap
        if (
            X is not None
            and len(X.select(pl.col(pl.Categorical)).columns) > 0
            and kwargs.get("fit_intercept") is True
        ):
            raise ValueError(
                "Dummy variable trap! Must set `fit_intercept=False` if X contains categorical columns."
            )

        regress = _linear_model(**kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg


@forecaster
def lasso(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Lasso autoregressive model."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _lasso(**kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg


@forecaster
def ridge(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Ridge autoregressive model."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _ridge(**kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg


@forecaster
def elastic_net(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Elastic net autoregressive model."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _elastic_net(**kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg
