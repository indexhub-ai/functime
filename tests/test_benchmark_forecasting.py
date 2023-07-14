import logging
from timeit import default_timer

import pytest


# 6 mins timeout
@pytest.mark.benchmark
def test_mlforecast_m4(pd_m4_dataset, benchmark):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from sklearn.linear_model import LinearRegression

    def fit_predict():
        y_train, _, fh = pd_m4_dataset
        entity_col, time_col = y_train.index.names
        model = MLForecast(
            models=[LinearRegression(fit_intercept=True)],
            lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            num_threads=cpu_count(),
        )
        model.fit(
            y_train,
            id_col=entity_col,
            time_col=time_col,
            target_col=y_train.columns[-1],
        )
        y_pred = model.predict(fh)
        return y_pred

    benchmark(fit_predict)


@pytest.mark.timeout(360)
@pytest.mark.benchmark
def test_darts_m4(pd_m4_dataset, benchmark):
    from darts import TimeSeries
    from darts.models import LinearRegressionModel

    def fit_predict():
        y_train, _, fh = pd_m4_dataset
        entity_col, time_col = y_train.index.names
        darts_y_train = TimeSeries.from_group_dataframe(
            y_train, group_cols=entity_col, time_col=time_col
        )
        model = LinearRegressionModel(lags=12)
        model.fit(darts_y_train)
        y_pred = model.predict(fh, series=darts_y_train)
        return y_pred

    benchmark(fit_predict)


@pytest.mark.benchmark
def test_mlforecast_m5(pd_m5_dataset, benchmark):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from sklearn.linear_model import LinearRegression

    def fit_predict():
        X_y_train, _, X_test, fh, freq = pd_m5_dataset
        entity_col, time_col = X_y_train.index.names
        model = MLForecast(
            models=[LinearRegression(fit_intercept=True)],
            lags=[1, 2, 3],
            num_threads=cpu_count(),
            freq=freq,
        )
        start = default_timer()
        model.fit(
            X_y_train.reset_index(),
            id_col=entity_col,
            time_col=time_col,
            target_col=X_y_train.columns[0],
            keep_last_n=24,
        )
        train_time = default_timer() - start
        logging.info("🛎️ mlforecast train time: %s", train_time)
        start = default_timer()
        y_pred = model.predict(fh, dynamic_dfs=[X_test])
        forecast_time = default_timer() - start
        logging.info("🛎️ mlforecast inference time: %s", forecast_time)
        return y_pred

    benchmark(fit_predict)
