import logging
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import modal
import polars as pl
from functime.metrics import mae

from functime.functions_stub import stub

try:
    from flaml import CFO, tune
    from flaml.tune.sample import Domain
except ImportError:
    pass


@stub.function(cpu=2.0, memory=10240, timeout=3600)
def evaluate_window(
    config: Mapping[str, Any],
    lags: int,
    test_size: int,
    max_horizons: int,
    strategy: str,
    freq: str,
    forecaster_cls: Callable,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    X_train: Optional[pl.DataFrame] = None,
    X_test: Optional[pl.DataFrame] = None,
) -> Union[float, pl.DataFrame]:
    # Train test split
    entity_col, time_col = y_train.columns[:2]
    # Fit
    config = config or {}
    forecaster = forecaster_cls(
        lags=lags, freq=freq, max_horizons=max_horizons, strategy=strategy, **config
    )
    try:
        forecaster.fit(y=y_train, X=X_train)
        y_pred = forecaster.predict(fh=test_size, X=X_test)
        # NOTE: Defensive match of entity, time indices
        # Need to do this for example if the train set is "1i", but the
        # test set starts from 1,2,3,...,fh
        # Assumes y_test and y_pred align up
        y_test = y_test.sort([entity_col, time_col])
        y_pred = y_pred.sort([entity_col, time_col])
        y_test = y_test.with_columns(**{time_col: y_pred.get_column(time_col)})
        score = mae(y_true=y_test, y_pred=y_pred).get_column("mae").mean()
        res = {"score": score}
    except ValueError as exc:
        # AttributeError: 'NoneType' object has no attribute 'last_result'
        # Root cause: y_preds are inf hence mae = inf
        logging.warning(
            "%s fit-predict failed with lags %s and parameters %s",
            forecaster_cls.func,
            lags,
            config,
            exc_info=exc,
        )
        res = None
    return res


@stub.function(memory=10240)
def evaluate_windows(
    config,
    lags: List[int],
    n_splits: int,
    test_size: int,
    max_horizons: int,
    strategy: str,
    freq: str,
    forecaster_cls: Callable,
    y_splits: Mapping[int, Tuple[pl.DataFrame, pl.DataFrame]],
    X_splits: Optional[Mapping[int, Tuple[pl.DataFrame, pl.DataFrame]]],
):
    # Get average mae across splits
    futures = []
    for i in range(n_splits):
        y_train, y_test = y_splits[i]
        X_train, X_test = X_splits[i] if X_splits is not None else None, None
        future = evaluate_window.spawn(
            y_train=y_train,
            y_test=y_test,
            X_train=X_train,
            X_test=X_test,
            config=config,
            lags=lags,
            test_size=test_size,
            max_horizons=max_horizons,
            strategy=strategy,
            freq=freq,
            forecaster_cls=forecaster_cls,
        )
        futures.append(future)
    results = [res for res in modal.functions.gather(*futures) if res is not None]
    scores = [res["score"] for res in results]
    score = None
    if len(scores) > 0:
        score = sum(scores) / len(scores)
    else:
        raise ValueError("Failed to evaluate every window")
    res = {"mae": score}
    return res


@stub.function(memory=10240)
def evaluate(
    lags: List[int],
    n_splits: int,
    time_budget: int,
    points_to_evaluate: List[Mapping[str, Any]],
    num_samples: int,
    low_cost_partial_config: Mapping[str, Any],
    test_size: int,
    max_horizons: int,
    strategy: str,
    freq: str,
    forecaster_cls: Callable,
    y_splits: Mapping[int, Tuple[pl.DataFrame, pl.DataFrame]],
    X_splits: Optional[Mapping[int, Tuple[pl.DataFrame, pl.DataFrame]]],
    search_space: Optional[Mapping[str, Domain]] = None,
):
    params = None
    if search_space is None:
        result = evaluate_windows(
            config=params,
            lags=lags,
            n_splits=n_splits,
            test_size=test_size,
            max_horizons=max_horizons,
            strategy=strategy,
            freq=freq,
            forecaster_cls=forecaster_cls,
            y_splits=y_splits,
            X_splits=X_splits,
        )
        score = result["mae"]
    else:
        tuner = tune.run(
            partial(
                evaluate_windows,
                lags=lags,
                n_splits=n_splits,
                test_size=test_size,
                max_horizons=max_horizons,
                strategy=strategy,
                freq=freq,
                forecaster_cls=forecaster_cls,
                y_splits=y_splits,
                X_splits=X_splits,
            ),
            config=search_space,
            metric="mae",
            mode="min",
            time_budget_s=time_budget,
            points_to_evaluate=points_to_evaluate,
            num_samples=num_samples,
            search_alg=CFO(low_cost_partial_config=low_cost_partial_config),
        )
        score = tuner.best_result["mae"]
        params = tuner.best_config
    return score
