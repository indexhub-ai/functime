import cloudpickle
import numpy as np
import polars as pl
import pytest
from functime.metrics import rmsse, smape
from sklearnex import patch_sklearn

from functime_backend.forecasting import (
    auto_elastic_net,
    auto_lightgbm,
    catboost,
    censored_model,
    elastic_net,
    knn,
    lightgbm,
    linear_model,
    xgboost,
)

patch_sklearn()

DEFAULT_LAGS = 12
DIRECT_KWARGS = {"max_horizons": 28, "strategy": "direct"}


@pytest.fixture(
    params=[
        (
            "catboost",
            lambda freq: catboost(lags=DEFAULT_LAGS, freq=freq, iterations=10),
        ),
        (
            "knn__direct",
            lambda freq: knn(
                lags=DEFAULT_LAGS, freq=freq, algorithm="kd_tree", **DIRECT_KWARGS
            ),
        ),
        ("knn", lambda freq: knn(lags=DEFAULT_LAGS, freq=freq)),
        (
            "lgbm__direct",
            lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, **DIRECT_KWARGS),
        ),
        (
            "lgbm",
            lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=10),
        ),
        (
            "linear__direct",
            lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, **DIRECT_KWARGS),
        ),
        ("linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq)),
        (
            "xgboost",
            lambda freq: xgboost(lags=DEFAULT_LAGS, freq=freq, n_estimators=10),
        ),
    ],
    ids=lambda model: model[0],
)
def forecaster(request):
    return request.param[1]


@pytest.fixture(
    params=[
        (
            "auto_elastic_net",
            lambda freq: auto_elastic_net(
                test_size=1, freq=freq, min_lags=3, max_lags=6
            ),
        ),
        (
            "auto_lgbm",
            lambda freq: auto_lightgbm(test_size=1, freq=freq, min_lags=3, max_lags=6),
        ),
    ],
    ids=lambda model: model[0],
)
def auto_forecaster(request):
    return request.param[1]


def test_forecaster_cloudpickle():
    y = pl.DataFrame(
        {
            "entity": ["a"] * 12 + ["b"] * 12,
            "time": list(range(12)) + list(range(12)),
            "target": [i + np.random.normal() for i in range(24)],
        }
    )
    forecaster = elastic_net(freq="1i", lags=3).fit(y=y)
    y_pred = forecaster.predict(fh=3)
    pickle = cloudpickle.dumps(forecaster)
    unpickled_forecaster = cloudpickle.loads(pickle)
    assert (
        smape(y_pred, unpickled_forecaster.predict(fh=3)).get_column("smape").sum()
        < 0.001
    )


def test_auto_cloudpickle():
    y = pl.DataFrame(
        {
            "entity": ["a"] * 12 + ["b"] * 12,
            "time": list(range(12)) + list(range(12)),
            "target": [i + np.random.normal() for i in range(24)],
        }
    )
    forecaster = auto_elastic_net(freq="1i", min_lags=3, max_lags=6).fit(y=y)
    y_pred = forecaster.predict(fh=3)
    pickle = cloudpickle.dumps(forecaster)
    unpickled_forecaster = cloudpickle.loads(pickle)
    assert (
        smape(y_pred, unpickled_forecaster.predict(fh=3)).get_column("smape").sum()
        < 0.001
    )


def test_forecaster_on_m4(forecaster, m4_dataset, benchmark):
    """Run global models against the M4 competition datasets and check overall RMSE
    (i.e. averaged across all time-series) is less than 2.
    """
    y_train, y_test, fh, freq = m4_dataset
    y_pred = benchmark(lambda: forecaster(freq=freq)(y=y_train, fh=fh))
    score = smape(y_test, y_pred).get_column("smape").mean()
    assert score < 0.3


def test_auto_on_m4(auto_forecaster, m4_dataset, benchmark):
    y_train, y_test, fh, freq = m4_dataset
    y_pred = benchmark(lambda: auto_forecaster(freq=freq)(y=y_train, fh=fh))
    score = smape(y_test, y_pred).get_column("smape").mean()
    assert score < 0.3


@pytest.mark.limit_memory("48GiB")
def test_forecaster_on_m5(forecaster, m5_dataset, benchmark):
    """Run global models against the M5 (Walmart) competition dataset and check
    overall RMSSE (i.e. averaged across all time-series) is less than 2.
    """
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = benchmark(
        lambda: forecaster(freq)(y=y_train, X=X_train, fh=fh, X_future=X_test)
    )
    score = rmsse(y_test, y_pred, y_train=y_train).get_column("rmsse").mean()
    assert score < 2


@pytest.mark.limit_memory("48GiB")
def test_auto_on_m5(auto_forecaster, m5_dataset, benchmark):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = benchmark(
        lambda: auto_forecaster(freq=freq)(y=y_train, X=X_train, fh=fh, X_future=X_test)
    )
    score = rmsse(y_test, y_pred, y_train=y_train).get_column("rmsse").mean()
    assert score < 2


@pytest.mark.parametrize("threshold", [5, 0])
def test_censored_model_on_m5(threshold, m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = censored_model(lags=3, threshold=threshold, freq=freq)(
        y=y_train, X=X_train, fh=fh, X_future=X_test
    )
    # Check column names
    assert y_pred.columns == [*y_train.columns[:3], "threshold_proba"]
    # Check dtypes
    assert y_pred.dtypes == [pl.Categorical, pl.Date, pl.Float64, pl.Float64]
    # Check score
    score = (
        rmsse(y_test, y_pred.select(y_train.columns[:3]), y_train=y_train)
        .get_column("rmsse")
        .mean()
    )
    assert score < 2
