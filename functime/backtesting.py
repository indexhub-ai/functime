from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import polars as pl

from functime.base import Forecaster
from functime.forecasting._reduction import (
    make_direct_reduction,
    make_reduction,
)


def _residualize_autoreg(
    y_train: pl.DataFrame,
    strategy: str,
    lags: int,
    max_horizons: int,
    artifacts: Mapping[str, Any],
    X_train: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    y_train = y_train.lazy().collect()
    idx_cols = y_train.columns[:2]

    def _score_recursive(regressor):
        X_y_train = make_reduction(lags=lags, y=y_train, X=X_train)
        X_cp_train = X_y_train.select(pl.all().exclude(target_col))
        y_pred_arr = regressor.predict(X_cp_train)
        # Check if censored model
        if isinstance(y_pred_arr, Tuple):
            y_pred_arr, _ = y_pred_arr  # forecast, probabilities
        y_pred_cp = X_cp_train.select(idx_cols).with_columns(
            pl.lit(y_pred_arr).alias("y_pred")
        )
        y_resids = y_pred_cp.join(
            y_train.rename({target_col: "y_train"}), on=idx_cols, how="left"
        ).select(*idx_cols, (pl.col("y_train") - pl.col("y_pred")).alias("y_resid"))
        return y_resids

    def _score_direct(regressors):
        X_y_train = make_direct_reduction(
            lags=lags, max_horizons=max_horizons, y=y_train, X=X_train
        )
        y_preds_cp = []
        for i in range(max_horizons):
            selected_lags = range(i + 1, lags + i + 1)
            feature_cols = [f"{target_col}__lag_{j}" for j in selected_lags]
            if X_train is not None:
                feature_cols += X_train.columns[2:]
            X_cp_train = X_y_train.select([*idx_cols, *feature_cols])
            y_pred_arr = regressors[i].predict(X_cp_train)
            # Check if censored model
            if isinstance(y_pred_arr, Tuple):
                y_pred_arr, _ = y_pred_arr  # forecast, probabilities
            y_preds_cp.append(y_pred_arr)
        # NOTE: we just naively take the mean across all direct predictions
        y_pred_arr = np.mean(y_preds_cp, axis=0)
        # Get y target values that match up with X_y_train (entity, time) index
        y_pred_cp = X_cp_train.select(idx_cols).with_columns(
            pl.lit(y_pred_arr).alias("y_pred")
        )
        y_resid = y_pred_cp.join(
            y_train.rename({target_col: "y_train"}), on=idx_cols, how="left"
        ).select(*idx_cols, (pl.col("y_train") - pl.col("y_pred")).alias("y_resid"))
        return y_resid

    target_col = y_train.columns[-1]
    if strategy == "ensemble":
        y_resid_recursive = _score_recursive(artifacts["recursive"]["regressor"])
        y_resid_direct = _score_direct(artifacts["direct"]["regressors"])
        y_resid = (
            y_resid_recursive.join(
                y_resid_direct, on=idx_cols, how="inner", suffix="__direct"
            )
            .with_columns((pl.col("y_resid") - pl.col("y_resid__direct")) / 2)
            .drop("y_resid__direct")
        )
    elif strategy == "recursive":
        y_resid = _score_recursive(artifacts["regressor"])
    else:
        y_resid = _score_direct(artifacts["regressors"])

    return y_resid


def _merge_autoreg_residuals(
    y: pl.DataFrame,
    y_resids: pl.DataFrame,
    strategy: str,
    lags: int,
    max_horizons: int,
    artifacts: Mapping[str, Any],
    X: Optional[pl.DataFrame] = None,
):
    y_resid = _residualize_autoreg(
        y_train=y,
        X_train=X,
        strategy=strategy,
        lags=lags,
        max_horizons=max_horizons,
        artifacts=artifacts,
    )
    last_split = y_resids.get_column("split").max() + 1
    y_resid = y_resid.with_columns(pl.lit(last_split).alias("split"))
    y_resids = pl.concat([y_resids, y_resid])
    return y_resids


def backtest(
    forecaster: Forecaster,
    y: pl.DataFrame,
    cv: Callable[[pl.DataFrame], Mapping[int, pl.DataFrame]],
    X: Optional[pl.DataFrame] = None,
    residualize: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    pl.enable_string_cache(True)
    entity_col = y.columns[0]
    y_splits = cv(y)
    X_splits = X if X is None else cv(X)
    y_preds = []
    y_resids = []
    for i in range(len(y_splits)):
        y_train, y_test = y_splits[i]
        fh = int(
            y_test.lazy()
            .select(pl.count() / pl.col(entity_col).n_unique())
            .collect()
            .item()
        )
        X_train, X_test = X_splits[i] if X is not None else None, None
        # Forecast
        model = forecaster.fit(y=y_train, X=X_train)
        y_pred = forecaster.predict(fh=fh, X=X_test)
        # Coerce split column names back into original names
        y_pred = y_pred.select(y_pred.columns[:3]).with_columns(
            pl.lit(i).alias("split")
        )
        # Append results
        y_preds.append(y_pred)
        if residualize:
            # Residuals
            y_resid = _residualize_autoreg(
                y_train=y_train,
                X_train=X_train,
                strategy=model.state.strategy,
                lags=forecaster.params["lags"],
                max_horizons=forecaster.params["max_horizons"],
                artifacts=model.state.artifacts,
            )
            y_resid = y_resid.with_columns(pl.lit(i).alias("split"))
            y_resids.append(y_resid)

    y_preds = pl.concat(y_preds)
    full_model = forecaster.fit(y=y, X=X)
    if residualize:
        y_resids = _merge_autoreg_residuals(
            y=y,
            X=X,
            y_resids=pl.concat(y_resids),
            strategy=full_model.state.strategy,
            lags=forecaster.params["lags"],
            max_horizons=forecaster.params["max_horizons"],
            artifacts=full_model.state.artifacts,
        )
        pl.enable_string_cache(False)
        return y_preds, y_resids
    pl.enable_string_cache(False)
    return y_preds
