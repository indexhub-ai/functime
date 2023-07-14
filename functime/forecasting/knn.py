from typing import Optional, Union

import polars as pl

from functime.base import forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.forecasting._ar import fit_autoreg, predict_autoreg
from functime.forecasting._regressors import StandardizedSklearnRegressor


def _knn(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.neighbors import KNeighborsRegressor

        regressor = StandardizedSklearnRegressor(
            estimator=KNeighborsRegressor(**kwargs, n_neighbors=1, n_jobs=-1),
        )
        return regressor.fit(X=X, y=y)

    return regress


@forecaster
def knn(
    freq: Union[str, None],
    lags: int,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Autoregressive k-nearest neighbors."""

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _knn(**kwargs)
        return fit_autoreg(
            regress=regress,
            lags=lags,
            y=y,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )

    return fit, predict_autoreg
