import json
from timeit import default_timer

import polars as pl

from functime.cross_validation import train_test_split
from functime.feature_extraction import add_calendar_effects, add_holiday_effects
from functime.forecasting import auto_lightgbm, lightgbm
from functime.metrics import mase

start_time = default_timer()

# Load data
y = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/commodities.parquet")
entity_col, time_col = y.columns[:2]
X = (
    y.select([entity_col, time_col])
    .pipe(add_calendar_effects(["month"]))
    .pipe(add_holiday_effects(country_codes=["US"], freq="1mo"))
    .collect()
)

print("🎯 Target variable (y):\n", y)
print("📉 Exogenous variables (X):\n", X)

# Train-test splits
test_size = 3
freq = "1mo"
y_train, y_test = train_test_split(test_size)(y)
X_train, X_test = train_test_split(test_size)(X)

# Univariate AutoML time-series fit with automated lags
# and hyperparameter tuning
auto_forecaster = auto_lightgbm(
    freq=freq,
    test_size=test_size,
    min_lags=20,
    max_lags=24,
    n_splits=3,
    time_budget=10,
)
auto_forecaster.fit(y=y_train)
# Predict
y_pred = auto_forecaster.predict(fh=test_size)
# Score
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
print("✅ Predictions (univariate):\n", y_pred.sort(entity_col))
print("💯 Scores (univariate):\n", scores)

# Retrieve AutoML "artifacts"
best_params = auto_forecaster.best_params
print(f"✨ Best parameters (y only):\n{json.dumps(best_params, indent=4)}")

# Multivariate non-AutoML
forecaster = lightgbm(**best_params)
forecaster.fit(y=y_train)
# Predict
y_pred = forecaster.predict(fh=test_size)
# Score
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)

print("✅ Predictions (with X):\n", y_pred.sort(entity_col))
print("💯 Scores (with X):\n", scores)

# "Direct" strategy forecasting
best_params["max_horizons"] = test_size  # Override max_horizons
best_params["strategy"] = "direct"  # Override strategy
# Predict using the "functional" API
y_pred = lightgbm(**best_params)(y=y_train, fh=test_size)

# "Ensemble" strategy forecasting
best_params["strategy"] = "ensemble"  # Override strategy
# Predict using the "functional" API
y_pred = lightgbm(**best_params)(y=y_train, fh=test_size)

elapsed_time = default_timer() - start_time
print(f"⏱️ Elapsed time: {elapsed_time}")
