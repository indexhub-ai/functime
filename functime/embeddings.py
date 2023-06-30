from base64 import b64decode
from io import BytesIO
from typing import Literal, Union, get_args

import numpy as np
import polars as pl
import pandas as pd
from functime.base.stub import _Stub

from functime.io.client import FunctimeH2Client

DF_TYPE = Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
EMBEDDING_MODELS = Literal["minirocket", "manyrocket"]
SUPPORTED_EMBEDDING_MODELS = get_args(EMBEDDING_MODELS)


class Embedder(_Stub):
    """Functime embedder client"""

    _stub_id: Union[str, None] = None

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "minirocket")
        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Model '{model}' not supported. Must be one of {SUPPORTED_EMBEDDING_MODELS}"
            )
        self.model: EMBEDDING_MODELS = model
        self.model_kwargs = kwargs

    def __call__(self, X: DF_TYPE) -> np.ndarray:
        return self.fit_transform(X=X)

    @classmethod
    def from_deployed(cls, stub_id: str, **kwargs):
        """Load a Embedder from a deployed estimator."""
        # Pull model metadata?
        response = ...
        response_json = response.json()
        kwargs.update(response_json["model_kwargs"])
        _cls = cls(**kwargs)
        _cls._stub_id = stub_id
        return _cls

    @property
    def stub_id(self) -> str:
        return self._stub_id

    @property
    def is_fitted(self) -> bool:
        return self._stub_id is not None

    @property
    def is_multivariate(self) -> bool:
        return self.is_fitted and self.n_variables > 1

    def fit(
        self,
        X: DF_TYPE,
        **kwargs,
    ):
        X = coerce_df_to_ndarray(X)

        kwargs = kwargs or {}
        arr_bytes = BytesIO(X.tobytes())
        dtype = str(X.dtype)
        shape = X.shape
        X_is_multivariate = len(shape) > 2
        if X_is_multivariate and self.model != "manyrocket":
            raise ValueError(
                f"Model '{self.model}' does not support multivariate time series. Please use ManyRocket instead."
            )

        n_instances = shape[0]
        self.n_variables = shape[1] if X_is_multivariate else 1
        self.n_timepoints = shape[-1]

        kwargs.update(
            {
                "model": self.model,
                "dtype": dtype,
                "n_instances": n_instances,
                "n_variables": self.n_variables,
                "n_timepoints": self.n_timepoints,
            }
        )

        with FunctimeH2Client(msg="Creating embeddings") as client:
            response = client.post(
                "/embed/fit",
                files={"X": arr_bytes},
                params=kwargs,
            )
        data = response.json()
        self._stub_id = data["stub_id"]
        return self._stub_id

    def transform(self, X: DF_TYPE, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Embedder has not been fitted yet.")
        X = coerce_df_to_ndarray(X)

        kwargs = kwargs or {}
        arr_bytes = BytesIO(X.tobytes())
        dtype = str(X.dtype)
        shape = X.shape
        X_is_multivariate = len(shape) > 2

        n_instances = shape[0]
        n_variables = shape[1] if X_is_multivariate else 1
        n_timepoints = shape[-1]

        if self.n_variables != n_variables or self.n_timepoints != n_timepoints:
            raise ValueError(
                f"Model was fitted with {self.n_variables} variables and {self.n_timepoints} "
                f"timepoints, but got {n_variables} variables and {n_timepoints} timepoints."
            )

        kwargs.update(
            {
                "model": self.model,
                "dtype": dtype,
                "n_instances": n_instances,
                "n_variables": n_variables,
                "n_timepoints": n_timepoints,
                "stub_id": self._stub_id,
            }
        )

        with FunctimeH2Client(msg="Creating embeddings") as client:
            response = client.post(
                "/embed/transform",
                files={"X": arr_bytes},
                params=kwargs,
            )
        data = response.json()
        # Reconstruct the np.array from the json
        emb = np.frombuffer(b64decode(data["embeddings"]), dtype=data["dtype"]).reshape(
            (data["n_instances"], data["embedding_length"])
        )
        return emb

    def fit_transform(
        self,
        X: DF_TYPE,
        **kwargs,
    ) -> np.ndarray:
        if self.is_fitted:
            return self.transform(X=X, **kwargs)

        X = coerce_df_to_ndarray(X)

        kwargs = kwargs or {}
        arr_bytes = BytesIO(X.tobytes())
        dtype = str(X.dtype)
        shape = X.shape
        print(f"shape: {shape}")
        X_is_multivatiate = len(shape) > 2
        if X_is_multivatiate and self.model != "manyrocket":
            raise ValueError(
                f"Model '{self.model}' does not support multivariate time series. Please use ManyRocket instead."
            )

        n_instances = shape[0]
        self.n_variables = 1 if not X_is_multivatiate else shape[1]
        self.n_timepoints = shape[-1]

        kwargs.update(
            {
                "model": self.model,
                "dtype": dtype,
                "n_instances": n_instances,
                "n_variables": self.n_variables,
                "n_timepoints": self.n_timepoints,
            }
        )

        with FunctimeH2Client(msg="Creating embeddings") as client:
            response = client.post(
                "/embed/fit_transform",
                files={"X": arr_bytes},
                params=kwargs,
            )
        data = response.json()
        # Reconstruct the np.array from the json
        emb = np.frombuffer(b64decode(data["embeddings"]), dtype=data["dtype"]).reshape(
            (data["n_instances"], data["embedding_length"])
        )
        self._stub_id = data["stub_id"]
        return emb


def coerce_df_to_ndarray(df: DF_TYPE) -> np.ndarray:
    if isinstance(df, np.ndarray):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_numpy()
    if isinstance(df, pl.LazyFrame):
        return df.collect().to_numpy()
    if isinstance(df, pd.DataFrame):
        return df.to_numpy()
    raise TypeError(f"Expected DataFrame, got {type(df)}")
