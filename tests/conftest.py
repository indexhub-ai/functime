import numpy as np
import pandas as pd
import polars as pl
import pytest
import logging
from functools import partial
from typing import List

import polars as pl
import pytest


@pytest.fixture(params=[250, 1000], ids=lambda x: f"n_periods({x})")
def n_periods(request):
    return request.param


@pytest.fixture(params=[50, 500], ids=lambda x: f"n_entities({x})")
def n_entities(request):
    return request.param


@pytest.fixture
def pd_X(n_periods, n_entities):
    """Return panel pd.DataFrame with sin, cos, and tan columns and time,
    entity multi-index. Used to benchmark polars vs pandas.
    """
    entity_idx = [f"x{i}" for i in range(n_entities)]
    time_idx = pd.date_range("2020-01-01", periods=n_periods, freq="1D", name="time")
    multi_row = [(entity, timestamp) for entity in entity_idx for timestamp in time_idx]
    idx = pd.MultiIndex.from_tuples(multi_row, names=["series_id", "time"])
    sin_x = np.sin(np.arange(0, n_entities * n_periods))
    cos_x = np.cos(np.arange(0, n_entities * n_periods))
    tan_x = np.tan(np.arange(0, n_entities * n_periods))
    X = pd.DataFrame({"open": sin_x, "close": cos_x, "volume": tan_x}, index=idx)
    return X.sort_index()


@pytest.fixture
def pd_y(pd_X):
    return pd_X.loc[:, "close"]


@pytest.fixture
def pl_y(pd_y):
    return pl.from_pandas(pd_y.reset_index()).lazy()


def prepare_m5_dataset(m5_train: pl.LazyFrame, m5_test: pl.LazyFrame):
    def filter_most_recent(
        X: pl.LazyFrame, max_eval_period: int, train_periods: int
    ) -> pl.LazyFrame:
        return X.filter(
            pl.col("d").str.slice(2).cast(pl.Int32) > max_eval_period - train_periods
        )

    def drop_leading_zeros(
        X: pl.LazyFrame, entity_col: str, target_col: str
    ) -> pl.LazyFrame:
        X_new = X.filter(
            ((pl.col(target_col) > 0).cast(pl.Int8).cummax().cast(pl.Boolean)).over(
                entity_col
            )
        )
        return X_new

    def preprocess(
        X: pl.LazyFrame,
        entity_col: str,
        sampled_entities: List[str],
        categorical_cols: List[str],
        boolean_cols: List[str],
    ) -> pl.LazyFrame:
        X_new = (
            X.select(
                pl.all().exclude(["d", "sell_price", "event_type_2", "event_name_2"])
            )  # Drop constant and unused columns
            .filter(pl.col(entity_col).is_in(sampled_entities))
            .with_columns(
                pl.col(categorical_cols).cast(pl.Utf8),
                pl.col(boolean_cols).cast(pl.Boolean),
            )
        )
        return X_new

    # Prepare M5 dataset
    # Specification
    sample_frac = 0.02  # 10% ~1.2 million rows
    entity_col = "id"
    time_col = "date"
    target_col = "quantity_sold"
    max_eval_period = 1914
    train_periods = 420
    categorical_cols = [
        "id",
        "state_id",
        "store_id",
        "dept_id",
        "cat_id",
        "item_id",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
    ]
    boolean_cols = ["snap_CA", "snap_TX", "snap_WI"]

    # Load train data and get entities sample
    # NOTE: Must sort, maintain order, and set seed to prevent flaky test
    sampled_entities = (
        m5_train.select(entity_col)
        .collect()
        .get_column(entity_col)
        .sort()
        .unique(maintain_order=True)
        .sample(fraction=sample_frac, seed=42)
    )

    preprocess_transform = partial(
        preprocess,
        entity_col=entity_col,
        sampled_entities=sampled_entities,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
    )

    X_y_train = (
        m5_train.pipe(
            filter_most_recent,
            max_eval_period=max_eval_period,
            train_periods=train_periods,
        )
        .pipe(drop_leading_zeros, entity_col=entity_col, target_col=target_col)
        .pipe(preprocess_transform)
        .collect()
    )
    X_y_test = m5_test.pipe(preprocess_transform).collect()

    # Train test split
    endog_cols = [entity_col, time_col, target_col]
    exog_cols = pl.all().exclude(target_col)
    y_train = X_y_train.select(endog_cols)
    X_train = X_y_train.select(exog_cols)
    y_test = X_y_test.select(endog_cols)
    X_test = X_y_test.select(exog_cols)

    return y_train, X_train, y_test, X_test


@pytest.fixture(
    params=[
        # ("m4_1h", 48),
        ("m4_1d", 14),
        ("m4_1w", 13),
        ("m4_1mo", 18),
        ("m4_3mo", 8),
        ("m4_1y", 6),
    ],
    ids=lambda x: "_".join(map(str, x)),
    scope="module",
)
def m4_dataset(request):
    def load_panel_data(path: str) -> pl.LazyFrame:
        return (
            pl.read_parquet(path)
            .pipe(lambda df: df.select(["series", "time", df.columns[-1]]))
            .with_columns(pl.col("series").str.replace(" ", ""))
            .sort(by="series")
        )

    dataset_id, fh = request.param
    freq = None  # I.e. test set starts at 1,2,3...,fh

    y_train = load_panel_data(f"data/{dataset_id}_train.parquet")
    y_test = load_panel_data(f"data/{dataset_id}_test.parquet")

    # Check m4 dataset RAM usage
    logging.info("y_train mem: %s", f'{y_train.estimated_size("mb"):.4f} mb')
    logging.info("y_test mem: %s", f'{y_test.estimated_size("mb"):.4f} mb')
    # Preview
    logging.info("y_train preview: %s", y_train)

    return y_train.lazy(), y_test.lazy(), fh, freq


@pytest.fixture
def m5_dataset():
    """M5 competition Walmart dataset grouped by stores."""

    # Specification
    fh = 28
    freq = "1d"

    # Load data
    y_train = pl.read_parquet(f"data/m5_y_train.parquet")
    X_train = pl.read_parquet(f"data/m5_X_train.parquet")
    y_test = pl.read_parquet(f"data/m5_y_test.parquet")
    X_test = pl.read_parquet(f"data/m5_X_test.parquet")
    # Check m5 dataset RAM usage
    logging.info("y_train mem: %s", f'{y_train.estimated_size("mb"):.4f} mb')
    logging.info("X_train mem: %s", f'{X_train.estimated_size("mb"):.4f} mb')
    logging.info("y_test mem: %s", f'{y_test.estimated_size("mb"):.4f} mb')
    logging.info("X_test mem: %s", f'{X_test.estimated_size("mb"):.4f} mb')

    # Preview
    logging.info("y_train preview: %s", y_train)
    logging.info("X_train preview: %s", X_train)

    return y_train.lazy(), X_train.lazy(), y_test.lazy(), X_test.lazy(), fh, freq


@pytest.fixture
def pd_m4_dataset(m4_dataset):
    y_train, y_test, fh = m4_dataset
    pd_y_train = y_train.collect().to_pandas()
    pd_y_test = y_test.collect().to_pandas()
    return pd_y_train, pd_y_test, fh


@pytest.fixture
def pd_m5_dataset(m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    entity_col, time_col = y_train.columns[:2]
    pd_y_train = y_train.collect().to_pandas().set_index([entity_col, time_col])
    pd_y_test = y_test.collect().to_pandas().set_index([entity_col, time_col])
    pd_X_train = X_train.collect().to_pandas().set_index([entity_col, time_col])
    pd_X_test = X_test.collect().to_pandas().set_index([entity_col, time_col])
    pd_X_y = pd_y_train.join(pd_X_train, how="left")
    return pd_X_y, pd_y_test, pd_X_test, fh, freq


# if __name__ == "__main__":
# y_train.collect().write_parquet("m5_y_train.parquet")
# X_train.collect().write_parquet("m5_X_train.parquet")
# y_test.collect().write_parquet("m5_y_test.parquet")
# X_test.collect().write_parquet("m5_X_test.parquet")
