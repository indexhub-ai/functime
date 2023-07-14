from typing import List

import polars as pl


def enbpi(
    y_pred: pl.LazyFrame,
    y_resid: pl.LazyFrame,
    alphas: List[float],
) -> pl.DataFrame:
    """Compute prediction intervals using ensemble batch prediction intervals (ENBPI)."""

    # 1. Group residuals by entity
    entity_col, time_col = y_pred.columns[:2]
    y_resid_grouped = y_resid.collect().groupby(entity_col, maintain_order=True)

    # 2. Forecast future prediction intervals: use constant residual quantile
    y_pred_quantiles = []
    for alpha in alphas:
        y_pred_quantile = (
            y_resid_grouped.quantile(alpha)
            .select([entity_col, y_resid.columns[-1]])
            .lazy()
            .join(y_pred.lazy(), how="left", on=entity_col)
            .select(
                [
                    entity_col,
                    time_col,
                    pl.col(y_pred.columns[-1]) + pl.col(y_resid.columns[-1]),
                    pl.lit(alpha).alias("quantile"),
                ]
            )
        )
        y_pred_quantiles.append(y_pred_quantile)

    y_pred_quantiles = (
        pl.concat(y_pred_quantiles).sort([entity_col, time_col]).collect()
    )
    return y_pred_quantiles


def conformalize(
    y_pred: pl.DataFrame,
    y_resids: pl.DataFrame,
    alphas: List[float],
) -> pl.DataFrame:
    """Compute prediction intervals using ensemble batch prediction intervals (ENBPI)."""

    y_pred = y_pred.lazy()
    y_resids = y_resids.lazy()

    # Aggregate bootstrapped residuals
    y_resid = y_resids.groupby(y_pred.columns[:2]).agg(
        pl.col(y_resids.columns[-2]).median()
    )
    y_pred_quantiles = enbpi(y_pred, y_resid, alphas)

    # Make alpha base 100
    y_pred_quantiles = y_pred_quantiles.with_columns(
        (pl.col("quantile") * 100).cast(pl.Int16)
    )

    return y_pred_quantiles
