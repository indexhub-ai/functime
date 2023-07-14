---
hide:
  - toc
---

`functime` supports both individual forecasters and AutoML forecasters.
AutoML forecasters uses `FLAML` to optimize both hyperparameters and number of lagged dependent variables.
`FLAML` is a [SOTA library](https://github.com/microsoft/FLAML) for AutoML and hyperparameter tuning using the CFO (Frugal Optimization for Cost-related Hyperparamters[^1]) algorithm.

All individual forecasters (e.g. `lasso` / `xgboost`) and AutoML forecasters (e.g. `auto_lasso` and `auto_xgboost`) implement the following API.

## `forecaster`

::: functime.base.forecaster.Forecaster

## `auto_forecaster`

::: functime.base.forecaster.AutoForecaster

[^1]: https://arxiv.org/abs/2005.01571
