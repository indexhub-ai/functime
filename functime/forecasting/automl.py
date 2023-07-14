from flaml import tune

from functime.base.forecaster import AutoForecaster
from functime.forecasting.knn import knn
from functime.forecasting.lightgbm import lightgbm
from functime.forecasting.linear import elastic_net, lasso, linear_model, ridge


class auto_lightgbm(AutoForecaster):
    DEFAULT_TREE_DEPTH = 8

    @property
    def model(self):
        return lightgbm

    @property
    def default_search_space(self):
        max_depth = self.kwargs.get("max_depth", 0)
        return {
            "reg_alpha": tune.loguniform(0.001, 20.0),
            "reg_lambda": tune.loguniform(0.001, 20.0),
            "num_leaves": tune.randint(
                2, 2**max_depth if max_depth > 0 else 2**self.DEFAULT_TREE_DEPTH
            ),
            "colsample_bytree": tune.uniform(0.4, 1.0),
            "subsample": tune.uniform(0.4, 1.0),
            "subsample_freq": tune.randint(1, 7),
            "min_child_samples": tune.qlograndint(5, 100, 5),
            "n_estimators": tune.qrandint(60, 400, 20),
        }

    @property
    def default_points_to_evaluate(self):
        return [
            {
                "num_leaves": 31,
                "colsample_bytree": 1.0,
                "subsample": 1.0,
                "min_child_samples": 20,
            }
        ]

    @property
    def low_cost_partial_config(self):
        return {"n_estimators": 50, "num_leaves": 2}


class auto_knn(AutoForecaster):
    @property
    def model(self):
        return knn

    @property
    def default_search_space(self):
        return {"leaf_size": tune.choice([30, 60, 120, 400])}

    @property
    def low_cost_partial_config(self):
        return {"leaf_size": 400}


class auto_linear_model(AutoForecaster):
    @property
    def model(self):
        return linear_model


class auto_lasso(AutoForecaster):
    @property
    def model(self):
        return lasso

    @property
    def default_search_space(self):
        return {
            "alpha": tune.loguniform(0.001, 20.0),
            "fit_intercept": tune.choice([True, False]),
        }

    @property
    def low_cost_partial_config(self):
        return {"alpha": 1.0}


class auto_ridge(AutoForecaster):
    @property
    def model(self):
        return ridge

    @property
    def default_search_space(self):
        return {
            "alpha": tune.loguniform(0.001, 20.0),
            "fit_intercept": tune.choice([True, False]),
        }

    @property
    def low_cost_partial_config(self):
        return {"alpha": 1.0}


class auto_elastic_net(AutoForecaster):
    @property
    def model(self):
        return elastic_net

    @property
    def default_search_space(self):
        return {
            "alpha": tune.loguniform(0.001, 20.0),
            "l1_ratio": tune.uniform(0, 1.0),
            "fit_intercept": tune.choice([True, False]),
        }

    @property
    def low_cost_partial_config(self):
        return {"alpha": 1.0}
