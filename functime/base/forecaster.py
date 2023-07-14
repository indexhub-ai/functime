import inspect
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, List, Mapping, Optional, Tuple, TypeVar, Union

import polars as pl
from functime.base.model import Model, ModelState
from functime.ranges import make_future_ranges
from typing_extensions import Literal, ParamSpec

# The parameters of the Model
P = ParamSpec("P")
# The return type of the esimator's curried function
R = Tuple[TypeVar("fit", bound=Callable), TypeVar("predict", bound=Callable)]

FORECAST_STRATEGIES = Optional[Literal["direct", "recursive", "naive"]]
DF_TYPE = Union[pl.LazyFrame, pl.DataFrame]


@dataclass(frozen=True)
class ForecastState(ModelState):
    target: str
    strategy: Optional[str] = "naive"
    features: Optional[List[str]] = None


class Forecaster(Model):
    """Autoregressive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    **kwargs : Mapping[str, Any]
        Additional keyword arguments passed into underlying sklearn-compatible estimator.
    """

    def __call__(
        self,
        y: DF_TYPE,
        fh: int,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
    ) -> pl.DataFrame:
        self.fit(y=y, X=X)
        return self.predict(fh=fh, X=X_future)

    @property
    def name(self):
        return f"{self.model.__name__}(strategy={self.params['strategy']})"

    def fit(self, y: DF_TYPE, X: Optional[DF_TYPE] = None):
        __fit, _ = self.func
        y: pl.LazyFrame = self._set_string_cache(y.lazy().collect()).lazy()
        if X is not None:
            if X.columns[0] == y.columns[0]:
                X = self._enforce_string_cache(X.lazy().collect())
            X = X.lazy()
        artifacts = __fit(y=y, X=X)
        cutoffs = y.groupby(y.columns[0]).agg(pl.col(y.columns[1]).max().alias("low"))
        artifacts["__cutoffs"] = cutoffs.collect(streaming=True)
        state = ForecastState(
            entity=y.columns[0],
            time=y.columns[1],
            artifacts=artifacts,
            target=y.columns[-1],
            strategy=self.params["strategy"] or "recursive",
            features=X.columns[2:] if X is not None else None,
        )
        self.state = state
        return self

    def predict(self, fh: int, X: Optional[DF_TYPE] = None) -> pl.DataFrame:
        _, __predict = self.func
        state = self.state
        entity = state.entity
        time = state.time
        target = state.target
        # Cutoffs cannot be lazy
        cutoffs: pl.DataFrame = state.artifacts["__cutoffs"]
        future_ranges = make_future_ranges(
            time_col=state.time,
            cutoffs=cutoffs,
            fh=fh,
            freq=self.params["freq"],
        )
        if X is not None:
            X = X.lazy()
            # Coerce X (can be panel / time series / cross sectional) into panel
            # and aggregate feature columns into lists
            has_entity = X.columns[0] == state.entity
            has_time = X.columns[1] == state.time

            if has_entity:
                X = self._enforce_string_cache(X.lazy().collect()).lazy()

            if has_entity and not has_time:
                X = future_ranges.lazy().join(X, on=entity, how="left")
            elif has_time and not has_entity:
                X = future_ranges.lazy().join(X, on=time, how="left")

            # NOTE: Unlike `y_lag` we DO NOT reshape exogenous features
            # into list columns. This is because .arr[i] with List[cat] does
            # not seem to support null values
            # Raises: ComputeError: cannot construct Categorical
            # from these categories, at least on of them is out of bounds
            X = X.select(pl.all().exclude(time)).lazy()
        y_pred_vals = __predict(state, fh=fh, X=X)
        y_pred_vals = y_pred_vals.rename(
            {x: y for x, y in zip(y_pred_vals.columns, [entity, target])}
        )
        y_pred = (
            future_ranges.lazy()
            .join(y_pred_vals.lazy(), on=entity)
            # Explode from wide arrs to long form
            # NOTE: Cannot use streaming here...
            # Causes change error "cannot append series, data types don't match
            .explode(pl.all().exclude(entity))
            .pipe(self._reset_string_cache)
            .collect()
        )
        return y_pred

    def backtest(
        self,
        y: DF_TYPE,
        X: Optional[pl.DataFrame],
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
    ):
        from functime.cross_validation import (
            expanding_window_split,
            sliding_window_split,
        )

        from functime.backtesting import backtest

        if strategy == "expanding":
            cv = expanding_window_split(
                test_size=test_size,
                step_size=step_size,
                n_splits=n_splits,
            )
        else:
            cv = sliding_window_split(
                test_size=test_size,
                step_size=step_size,
                n_splits=n_splits,
                window_size=window_size,
            )
        y_preds, y_resids = backtest(
            forecaster=self,
            y=y,
            X=X,
            cv=cv,
            residualize=True,
        )
        return y_preds, y_resids

    def conformalize(
        self,
        fh: int,
        y: pl.DataFrame,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
        alphas: Optional[List[float]] = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
        return_results: bool = False,
    ) -> pl.DataFrame:
        from functime.conformal import conformalize

        y_pred = self.predict(fh=fh, X=X_future)
        y_preds, y_resids = self.backtest(
            y=y,
            X=X,
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
            window_size=window_size,
            strategy=strategy,
        )
        y_pred = pl.concat(
            [
                y_pred,
                y_preds.groupby(y_pred.columns[:2]).agg(
                    pl.col(y_pred.columns[-1])
                    .median()
                    .cast(y_pred.schema[y_pred.columns[-1]])
                ),
            ]
        )
        y_pred_qnts = conformalize(y_pred, y_resids, alphas=alphas)
        if return_results:
            return y_pred, y_pred_qnts, y_preds, y_resids
        return y_pred_qnts


class AutoForecaster(Forecaster):
    """AutoML forecaster with automated hyperparameter tuning and lags selection.

    Parameters
    ----------
    freq : str
        Offset alias as dictated.
    min_lags : int
        Minimum number of lagged target values.
    max_lags : int
        Maximum number of lagged target values.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    test_size : int
        Number of lags.
    step_size : int
        Step size between backtest windows.
    n_splits : int
        Number of backtest splits.
    time_budget : int
        Maximum time budgeted to train each forecaster per window and set of hyperparameters.
    search_space : Optional[dict]
        Equivalent to `config` in [FLAML](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#search-space)
    points_to_evaluate : Optional[dict]
        Equivalent to `points_to_evaluate` in [FLAML](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#warm-start)
    num_samples : int
        Number of hyper-parameter sets to test. -1 means unlimited (until `time_budget` is exhausted.)
    **kwargs : Mapping[str, Any]
        Additional keyword arguments passed into underlying sklearn-compatible estimator.
    """

    def __init__(
        self,
        # NOTE: MUST EXPLICITLY SPECIFIC FREQ IN ORDER FOR
        # CROSS-VALIDATION y_pred and y_test time index to match up
        freq: Union[str, None],
        min_lags: int = 3,
        max_lags: int = 12,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        time_budget: int = 5,
        search_space: Optional[Mapping[str, Any]] = None,
        points_to_evaluate: Optional[Mapping[str, Any]] = None,
        num_samples: int = -1,
        **kwargs,
    ):
        self.freq = freq
        self.min_lags = min_lags
        self.max_lags = max_lags
        self.max_horizons = max_horizons
        self.strategy = strategy
        self.test_size = test_size
        self.step_size = step_size
        self.n_splits = n_splits
        self.time_budget = time_budget
        self.search_space = search_space
        self.points_to_evaluate = points_to_evaluate
        self.num_samples = num_samples
        self.kwargs = kwargs

    @property
    def args(self):
        return (
            self.freq,
            self.min_lags,
            self.max_lags,
            self.max_horizons,
            self.strategy,
            self.test_size,
            self.step_size,
            self.n_splits,
            self.time_budget,
            self.search_space,
            self.points_to_evaluate,
            self.num_samples,
        )

    @property
    def params(self):
        kwargs = self.kwargs
        sig = inspect.signature(self.__init__)
        model_params = sig.parameters
        model_args = list(model_params.keys())
        params = {
            **{
                k: kwargs.get(k, v.default)
                for k, v in model_params.items()
                if k != "kwargs"
            },
            **{model_args[i]: p for i, p in enumerate(self.args)},
        }
        return params

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    def name(self):
        return f"auto_{self.model.__name__}"

    @property
    def default_search_space(self):
        return None

    @property
    def default_points_to_evaluate(self):
        return None

    @property
    def low_cost_partial_config(self):
        return None

    @property
    def func(self):
        from functime.forecasting._ar import fit_cv, predict_autoreg

        def fit(y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
            return fit_cv(
                y=y,
                X=X,
                forecaster_cls=partial(self.model, **self.kwargs),
                freq=self.freq,
                min_lags=self.min_lags,
                max_lags=self.max_lags,
                max_horizons=self.max_horizons,
                strategy=self.strategy,
                test_size=self.test_size,
                step_size=self.step_size,
                n_splits=self.n_splits,
                time_budget=self.time_budget,
                search_space=self.search_space or self.default_search_space,
                points_to_evaluate=self.points_to_evaluate
                or self.default_points_to_evaluate,
                num_samples=self.num_samples,
                low_cost_partial_config=self.low_cost_partial_config,
            )

        return fit, predict_autoreg

    def backtest(
        self,
        y: DF_TYPE,
        X: Optional[pl.DataFrame],
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
    ):
        # Get base forecaster with fixed best params
        forecaster_cls = self.model
        if self.state is None:
            raise ValueError("Must `.fit` AutoForecaster before `.backtest`")
        best_params = self.state.artifacts["best_params"]
        forecaster = forecaster_cls(**best_params)
        y_preds, y_resids = forecaster.backtest(
            y=y,
            X=X,
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
            window_size=window_size,
            strategy=strategy,
        )
        return y_preds, y_resids


def forecaster(model: Callable[P, R]):
    @wraps(model)
    def _forecaster(*args: P.args, **kwargs: P.kwargs) -> Forecaster:
        return Forecaster(model, *args, **kwargs)

    return _forecaster
