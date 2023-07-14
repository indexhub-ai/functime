from typing import Callable, Optional, Union

import polars as pl

from functime.base import forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.forecasting._ar import fit_autoreg, predict_autoreg
from functime.forecasting._reduction import make_reduction
from functime.forecasting._regressors import CensoredRegressor
from functime.forecasting.linear import _linear_model


def auto_classify(X: pl.DataFrame, y: pl.DataFrame):
    from flaml import AutoML

    tuner = AutoML(
        estimator_list=["lgbm", "lrl1"],
        time_budget=30,
        metric="f1",
        task="classification",
    )

    tuner.fit(
        X_train=X.select(X.columns[2:]).to_pandas(),
        y_train=y.get_column(y.columns[-1]).to_pandas(),
    )
    estimator = tuner.model.estimator
    return estimator


def default_classify(X: pl.DataFrame, y: pl.DataFrame):
    from sklearn.ensemble import RandomForestClassifier

    estimator = RandomForestClassifier()
    estimator.fit(
        X=X.select(X.columns[2:]).to_numpy(),
        y=y.get_column(y.columns[-1]).to_numpy(),
    )
    return estimator


@forecaster
def censored_model(
    freq: Union[str, None],
    lags: int,
    threshold: Union[int, float],
    regress: Optional[Callable] = None,
    classify: Optional[Callable] = None,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Censored model for forecasting data in which the target variable is censored above or below a certain threshold."""

    regress = _linear_model() if regress is None else regress(**kwargs)
    classify = default_classify if classify is None else classify

    def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        # 1. Fit classifier
        target_col = y.columns[-1]
        X_y_final = (
            make_reduction(lags=lags, y=y, X=X)
            .with_columns(
                pl.when(pl.col(target_col) > threshold)
                .then(1)
                .otherwise(0)
                .alias(target_col)
            )
            .lazy()
        )
        X_final, y_final = pl.collect_all(
            [
                X_y_final.select(pl.all().exclude(target_col)),
                X_y_final.select([*X_y_final.columns[:2], target_col]),
            ]
        )
        fitted_classifier = classify(X=X_final, y=y_final)
        # 2. Fit forecast model on non-zeros
        censored_regressor = CensoredRegressor(
            threshold=threshold,
            regress=regress,
            predict_proba=fitted_classifier.predict_proba,
        )
        forecast_artifacts = fit_autoreg(
            regress=censored_regressor.fit,
            lags=lags,
            y=y,
            X=X,
            max_horizons=max_horizons,
            strategy=strategy,
        )
        # 3. Collect artifacts
        artifacts = {"classifier": fitted_classifier, **forecast_artifacts}
        return artifacts

    return fit, predict_autoreg


def zero_inflated_model(
    freq: Union[str, None],
    lags: int,
    regress: Optional[Callable] = None,
    classify: Optional[Callable] = None,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    **kwargs
):
    """Zero inflated forecaster."""
    return censored_model(
        freq=freq,
        lags=lags,
        threshold=0,
        regress=regress,
        classify=classify,
        max_horizons=max_horizons,
        strategy=strategy,
        **kwargs
    )
