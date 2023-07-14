import logging
import os
from typing import Any, Callable, List, Mapping, Optional, Type, Union

import modal
import numpy as np
import polars as pl
from functime.cross_validation import expanding_window_split

from functime.base import Forecaster
from functime.base.forecaster import FORECAST_STRATEGIES, ForecastState
from functime.forecasting._reduction import (
    make_direct_reduction,
    make_reduction,
    make_y_lag,
)

try:
    from flaml.tune.sample import Domain
except ImportError:
    pass


def fit_recursive(
    regress: Callable[[pl.LazyFrame, pl.LazyFrame], Any],
    lags: int,
    y: pl.LazyFrame,
    X: Optional[pl.LazyFrame] = None,
) -> Mapping[str, Any]:
    # 1. Impose AR structure
    target_col = y.columns[-1]
    X_y_final = make_reduction(lags=lags, y=y, X=X).lazy()
    X_final, y_final = pl.collect_all(
        [
            X_y_final.select(pl.all().exclude(target_col)),
            X_y_final.select([*X_y_final.columns[:2], target_col]),
        ]
    )
    # 2. Fit
    fitted_model = regress(X=X_final, y=y_final)
    # 3. Collect artifacts
    y_lag = make_y_lag(X_y_final, target_col=y.columns[-1], lags=lags)
    artifacts = {
        "regressor": fitted_model,
        "y_lag": y_lag.collect(streaming=True),
    }
    return artifacts


def fit_direct(
    regress: Callable[[pl.LazyFrame, pl.LazyFrame], Any],
    lags: int,
    max_horizons: int,
    y: pl.LazyFrame,
    X: Optional[pl.LazyFrame] = None,
) -> Mapping[str, Any]:
    idx_cols = y.columns[:2]
    target_col = y.columns[-1]
    feature_cols = X.columns[2:] if X is not None else []
    # 1. Impose AR structure
    X_y_final = make_direct_reduction(lags=lags, max_horizons=max_horizons, y=y, X=X)
    # 2. Fit
    # TODO: Modalize
    fitted_models = []
    for i in range(1, max_horizons + 1):
        selected_lags = range(i, lags + i)
        lag_cols = [f"{target_col}__lag_{j}" for j in selected_lags]
        X_final = X_y_final.select([*idx_cols, *lag_cols, *feature_cols])
        y_final = X_y_final.select([*idx_cols, target_col])
        fitted_model = regress(X=X_final, y=y_final)
        fitted_models.append(fitted_model)
    # 3. Collect artifacts
    y_lag = make_y_lag(X_y_final, target_col=y.columns[-1], lags=lags + max_horizons)
    artifacts = {
        "regressors": fitted_models,
        "y_lag": y_lag.collect(streaming=True),
    }
    return artifacts


def fit_autoreg(
    regress: Callable[[pl.LazyFrame, pl.LazyFrame], Any],
    lags: int,
    y: Union[pl.DataFrame, pl.LazyFrame],
    X: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
) -> Mapping[str, Any]:
    y = y.lazy()
    X = X.lazy() if X is not None else X
    strategy = strategy or "recursive"
    if strategy in ["direct", "ensemble"] and max_horizons is None:
        raise ValueError(
            "If `strategy` is set as 'direct' or 'ensemble', then `max_horizons` must be set"
            " in the forecaster's kwargs upon initialization."
        )
    if strategy == "recursive":
        artifacts = fit_recursive(regress=regress, lags=lags, y=y, X=X)
    elif strategy == "direct":
        artifacts = fit_direct(
            regress=regress, lags=lags, max_horizons=max_horizons, y=y, X=X
        )
    elif strategy == "ensemble":
        artifacts = {
            "recursive": fit_recursive(regress=regress, lags=lags, y=y, X=X),
            "direct": fit_direct(
                regress=regress, lags=lags, max_horizons=max_horizons, y=y, X=X
            ),
        }
    else:
        raise ValueError(f"Cannot recognize `strategy` '{strategy}'")
    return artifacts


def fit_cv(  # noqa: Ruff too complex
    y: pl.LazyFrame,
    forecaster_cls: Type[Forecaster],
    freq: Union[str, None],
    min_lags: int = 3,
    max_lags: int = 12,
    max_horizons: Optional[int] = None,
    strategy: FORECAST_STRATEGIES = None,
    test_size: int = 1,
    step_size: int = 1,
    n_splits: int = 5,
    time_budget: int = 5,
    search_space: Optional[Mapping[str, Domain]] = None,
    points_to_evaluate: Optional[List[Mapping[str, Any]]] = None,
    low_cost_partial_config: Optional[Mapping[str, Any]] = None,
    num_samples: int = -1,
    cv: Optional[
        Callable[[pl.LazyFrame, bool, bool], Union[pl.LazyFrame, pl.DataFrame]]
    ] = None,
    X: Optional[pl.LazyFrame] = None,
    **kwargs,
) -> Mapping[str, Any]:
    # Set defaults
    strategy = strategy or "recursive"
    # Prepare CV splits query plan i.e. LazyFrames
    cv = cv or expanding_window_split(
        test_size=test_size, n_splits=n_splits, step_size=step_size, eager=True
    )
    y_splits = cv(y)
    X_splits = X if X is None else cv(X)

    # Test each lag
    best_lags = None
    best_params = None
    scores_path = []
    lags_path = list(range(min_lags, max_lags + 1))
    env_prefix = os.environ.get("FUNCTIME__ENV_NAME", "dev")
    evaluate = modal.Function.lookup(f"{env_prefix}-functime-functions", "evaluate")
    scores_path = [
        score
        for score in evaluate.map(
            lags_path,
            kwargs={
                "n_splits": n_splits,
                "time_budget": time_budget,
                "points_to_evaluate": points_to_evaluate,
                "num_samples": num_samples,
                "low_cost_partial_config": low_cost_partial_config,
                "search_space": search_space,
                "test_size": test_size,
                "max_horizons": max_horizons,
                "strategy": strategy,
                "freq": freq,
                "forecaster_cls": forecaster_cls,
                "y_splits": y_splits,
                "X_splits": X_splits,
            },
            order_outputs=True,
        )
    ]
    best_idx = np.argmin(scores_path)
    best_score = scores_path[best_idx]
    best_lags = lags_path[best_idx]

    # Refit
    best_params = best_params or {}
    best_params = {
        "freq": freq,
        **best_params,
        "max_horizons": max_horizons,
        "strategy": strategy,
        **kwargs,
    }
    best_params["lags"] = best_lags
    logging.info("✅ Found `best_params` %s", best_params)
    best_model = forecaster_cls(**best_params)
    best_model.fit(y=y, X=X)
    # Prepare artifacts
    # TODO: Investigate ensembling across hyperparameter sets
    # Ref: https://arxiv.org/abs/2006.13570
    artifacts = {
        **best_model.state.artifacts,
        "best_score": best_score,
        "best_params": best_params,
        "lags_path": lags_path,
        "scores_path": scores_path,
    }

    return artifacts


# NOTE: REMEMBER exogenous X DOES NOT HAVE TIME_COL
# (values are aggregated into list before being passed into predict)


def predict_recursive(
    state: ForecastState,
    fh: int,
    X: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    artifacts = state.artifacts
    if "recursive" in artifacts.keys():
        artifacts = state.artifacts["recursive"]
    regressor = artifacts["regressor"]
    entity_col = state.entity
    y_lag: pl.DataFrame = artifacts["y_lag"].sort(entity_col)
    X = X.sort(entity_col) if X is not None else X
    lag_cols = y_lag.columns[2:]
    lead_col = lag_cols[0]

    def _get_x_y_slice(y_lag: pl.DataFrame, i: int):
        x_y_slice = y_lag.select(
            [entity_col, pl.all().exclude(entity_col).list.get(-1)]
        )
        if X is not None:
            x = X.groupby(entity_col, maintain_order=True).agg(
                pl.all().exclude(entity_col).take(i)
            )
            x_y_slice = x_y_slice.join(x, on=entity_col, how="left")
        return x_y_slice

    is_censored = getattr(regressor, "is_censored", False)
    weights = np.zeros((fh, len(y_lag))) if is_censored else None

    for i in range(fh):
        # 1. Get most recent features
        x_y_slice = _get_x_y_slice(y_lag=y_lag, i=i)
        # 2. Predict
        y_pred_i = regressor.predict(x_y_slice)
        if is_censored:
            y_pred_i, weights_i = y_pred_i
            weights[i] = weights_i
        # 3. Update AR structure
        y_shifted = [
            pl.col(lag_cols[i]).alias(lag_cols[i + 1])
            for i in range(0, len(lag_cols) - 1)
        ]
        y_new = pl.col(lead_col).list.concat(pl.Series(y_pred_i))
        y_lag = y_lag.with_columns([y_new, *y_shifted])

    pred_cols = [entity_col, pl.col(lead_col).list.tail(fh).alias(state.target)]
    y_pred = y_lag.select(pred_cols)

    if is_censored:
        weights = pl.DataFrame(np.stack(weights, axis=1)).select(
            pl.concat_list(pl.all()).alias("threshold_proba")
        )
        y_pred = pl.concat([y_pred, weights], how="horizontal")

    return y_pred


# NOTE: REMEMBER exogenous X DOES NOT HAVE TIME_COL
# (values are aggregated into list before being passed into predict)


def predict_direct(
    state: ForecastState, fh: int, X: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    entity_col = state.entity
    time_col = state.time
    target_col = state.target
    X_feature_cols = X.columns[1:] if X is not None else []
    artifacts = state.artifacts
    if "direct" in artifacts.keys():
        artifacts = state.artifacts["direct"]
    regressors = artifacts["regressors"]
    max_horizons = len(regressors)

    if fh > max_horizons:
        raise ValueError(
            "`fh` must be less than or equal to `max_horizons` in model parameters."
            f" Expected `fh <= {max_horizons}`, got `{fh}`."
        )

    y_lag: pl.DataFrame = artifacts["y_lag"].sort(entity_col)
    X = X.sort(entity_col) if X is not None else X
    lags = (y_lag.width - 1) - max_horizons

    n_entities = len(y_lag)
    y_pred = np.empty((n_entities, fh))
    is_censored = getattr(regressors[0], "predict_proba", None)
    weights = np.zeros((fh, n_entities)) if is_censored else None

    for i in range(fh):
        selected_lags = range(i + 1, lags + i)
        lag_cols = [pl.col(f"{target_col}__lag_{j}").list.get(i) for j in selected_lags]
        x = y_lag.select([entity_col, pl.col(time_col).list.get(i), *lag_cols])
        if X is not None:
            x_slice = X.groupby(entity_col, maintain_order=True).agg(
                pl.col(X_feature_cols).take(i)
            )
            x = x.join(x_slice, on=entity_col, how="left")
        # Predict
        y_pred_i = regressors[i].predict(x)
        # Censored forecast adjustment
        if is_censored:
            y_pred_i, weights_i = y_pred_i
            weights[i] = weights_i
        y_pred[:, i] = y_pred_i

    y_pred = (
        pl.DataFrame(y_pred)
        .select(pl.concat_list(pl.all()).alias(target_col))
        .with_columns(y_lag.get_column(entity_col))
        .select([entity_col, target_col])
    )
    if is_censored:
        weights = pl.DataFrame(np.stack(weights, axis=1)).select(
            pl.concat_list(pl.all()).alias("threshold_proba")
        )
        y_pred = pl.concat([y_pred, weights], how="horizontal")

    return y_pred


# NOTE: REMEMBER exogenous X DOES NOT HAVE TIME_COL
# (values are aggregated into list before being passed into predict)


def predict_autoreg(
    state: ForecastState,
    fh: int,
    X: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
) -> pl.DataFrame:
    strategy = state.strategy
    predict_kwargs = {
        "state": state,
        "fh": fh,
        "X": X.lazy().collect() if X is not None else None,
    }
    if strategy == "recursive":
        y_pred = predict_recursive(**predict_kwargs)
    elif strategy == "direct":
        y_pred = predict_direct(**predict_kwargs)
    elif strategy == "ensemble":
        target_col = state.target
        y_pred_rec = predict_recursive(**predict_kwargs).rename(
            {target_col: "recursive"}
        )
        y_pred_dir = predict_direct(**predict_kwargs).rename({target_col: "direct"})
        y_pred = (
            y_pred_rec.join(y_pred_dir, on=state.entity)
            .explode(["recursive", "direct"])
            .select(
                [
                    state.entity,
                    ((pl.col("recursive") + pl.col("direct")) / 2).alias(target_col),
                ]
            )
            .groupby(state.entity)
            .agg(target_col)
        )
    else:
        raise ValueError(f"Cannot recognize `strategy` '{strategy}'")
    return y_pred
