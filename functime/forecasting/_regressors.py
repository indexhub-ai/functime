"""
Fit-predict regressors with special needs.
"""

from typing import Callable, Optional, Union

import numpy as np
import polars as pl
from functime.preprocessing import PL_NUMERIC_COLS
from typing_extensions import Literal


def _X_to_numpy(X: pl.DataFrame) -> np.ndarray:
    X_arr = X.select(pl.col(X.columns[2:])).to_numpy().astype(np.float32)
    return X_arr


def _y_to_numpy(y: pl.DataFrame) -> np.ndarray:
    return y.get_column(y.columns[-1]).to_numpy().astype(np.float32)


class GradientBoostedTreeRegressor:
    def __init__(
        self,
        regress,
        weight_transform: Optional[Callable] = None,
        fit_dtype: Literal["numpy", "arrow"] = None,
        predict_dtype: Union[Literal["numpy", "arrow"], Callable] = None,
    ):
        self.regress = regress
        self.regressor = None
        self.weight_transform = weight_transform
        self.fit_dtype = fit_dtype or "numpy"
        self.predict_dtype = predict_dtype or "numpy"
        self.label_to_cat = {}

    def _preproc_X(self, X: pl.DataFrame) -> pl.DataFrame:
        entity_col = X.columns[0]
        X_new = X.with_columns(
            pl.col([pl.Categorical, pl.Boolean]).exclude(entity_col).to_physical()
        )
        return X_new

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):

        weight_transform = self.weight_transform
        sample_weight = None
        if weight_transform is not None:
            sample_weight = y.pipe(weight_transform)

        X = self._preproc_X(X)

        if self.fit_dtype == "numpy":
            X_coerced = _X_to_numpy(X)
            y_coerced = _y_to_numpy(X)
        elif self.fit_dtype == "arrow":
            X_coerced = X.to_arrow()
            y_coerced = y.to_arrow()
        else:
            raise ValueError(f"`fit_dtype` not supported: {self.fit_dtype}")

        self.regressor = self.regress(
            X=X_coerced, y=y_coerced, sample_weight=sample_weight
        )
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X = self._preproc_X(X)
        if self.predict_dtype == "numpy":
            X_coerced = _X_to_numpy(X)
        elif self.predict_dtype == "arrow":
            X_coerced = X.to_arrow
        elif isinstance(self.predict_dtype, Callable):
            X_coerced = self.predict_dtype(X)
        y_pred = self.regressor.predict(X_coerced)
        return y_pred


class StandardizedSklearnRegressor:
    def __init__(self, estimator):
        self.estimator = estimator
        self.pipeline = None

    def _preproc_X(self, X: pl.DataFrame):
        entity_col, time_col = X.columns[:2]
        X_new = X.select(
            [
                entity_col,
                time_col,
                PL_NUMERIC_COLS(entity_col, time_col),
                pl.col(pl.Categorical).exclude(entity_col).to_physical(),
                pl.col(pl.Boolean).cast(pl.Int8),
            ]
        )
        return X_new

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

        # Get column names
        entity_col, time_col = X.columns[:2]
        numeric_cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns
        categorical_cols = X.select(pl.col(pl.Categorical).exclude(entity_col)).columns
        boolean_cols = X.select(pl.col(pl.Boolean)).columns
        # Feature sizes
        n_numeric_cols = len(numeric_cols)
        n_categorical_cols = len(categorical_cols)
        n_boolean_cols = len(boolean_cols)
        # Feature slices
        numeric_idx = slice(0, n_numeric_cols)
        categorical_idx = slice(n_numeric_cols, n_numeric_cols + n_categorical_cols)
        boolean_idx = slice(
            n_numeric_cols + n_categorical_cols,
            n_numeric_cols + n_categorical_cols + n_boolean_cols,
        )
        # Defensive reordering X and cast boolean to 0, 1
        X = self._preproc_X(X)
        scaler = MaxAbsScaler()
        ohe = OneHotEncoder(
            drop=None, dtype=np.int8, sparse_output=False, handle_unknown="ignore"
        )

        transformers = [("numeric", scaler, numeric_idx)]
        if len(categorical_cols) > 1:
            transformers += [("categorical", ohe, categorical_idx)]
        if len(boolean_cols) > 1:
            transformers += [("boolean", "passthrough", boolean_idx)]

        transformer = ColumnTransformer(transformers=transformers)
        steps = [("transformer", transformer), ("regressor", self.estimator)]

        # Fit pipeline
        pipeline = Pipeline(steps=steps)
        self.pipeline = pipeline.fit(X=_X_to_numpy(X), y=_y_to_numpy(y))
        return self

    def predict(self, X: pl.DataFrame):
        # Defensive reordering X and cast boolean to 0, 1
        X = self._preproc_X(X)
        y_pred = self.pipeline.predict(_X_to_numpy(X))
        return y_pred


class CensoredRegressor:
    def __init__(
        self,
        threshold: Union[int, float],
        regress,
        predict_proba,
    ):
        self.threshold = threshold
        self.regress = regress
        self.predict_proba = predict_proba
        self.regressors = None

    @property
    def is_censored(self):
        return True

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        idx_cols = X.columns[:2]
        target_col = y.columns[-1]
        threshold = self.threshold
        X_y_above = X.join(y, on=idx_cols).filter(pl.col(target_col) > threshold)
        y_above = X_y_above.select([*idx_cols, target_col])
        X_above = X_y_above.select(pl.all().exclude(target_col))
        if threshold == 0:
            self.regressors = self.regress(X_above, y_above), None
        else:
            X_y_below = X.join(y, on=idx_cols).filter(pl.col(target_col) <= threshold)
            y_below = X_y_below.select([*idx_cols, target_col])
            X_below = X_y_below.select(pl.all().exclude(target_col))
            fitted_model_above = self.regress(X_above, y_above)
            fitted_model_below = self.regress(X_below, y_below)
            self.regressors = fitted_model_above, fitted_model_below
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        weights = self.predict_proba(_X_to_numpy(X))
        regress_above, regress_below = self.regressors
        y_pred = weights[:, 1] * regress_above.predict(X)
        if abs(self.threshold) > 0:
            y_pred += weights[:, 0] * regress_below.predict(X)
        return y_pred, weights[:, 1]
