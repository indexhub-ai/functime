# functime

![functime](img/banner.png)

## Production-ready time series models

**functime** is a machine learning library for time-series predictions that [just works](https://www.functime.ai/).

- **Fully-featured:** Powerful and easy-to-use API for [AutoML forecasting](#forecasting-highlights) and [temporal embeddings](#embeddings-highlights) (classification, anomaly detection, and clustering)
- **Fast:** Forecast and classify [100,000 time series](#global-forecasting) in seconds *on your laptop*
- **Cloud-native:** Instantly run, deploy, and serve predictive time-series models
- **Efficient:** Embarressingly parallel feature engineering using [Polars](https://www.pola.rs/) *
- **Battle-tested:** Algorithms that deliver real business impact and win competitions

## Installation

Check out this [guide](installation.md) to install functime. Requires Python 3.8+.

## Forecasting

Point and probablistic forecasts using machine learning.
Includes utilities to support the full forecasting lifecycle:
preprocessing, feature extraction, time-series cross-validation / splitters, backtesting, AutoML, hyperparameter tuning, and scoring.

- Every forecaster supports **exogenous features**
- **Backtesting** with expanding window and sliding window splitters
- **Automated lags and hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- **Probablistic forecasts** via quantile regression and conformal prediction
- **Forecast metrics** (e.g. MASE, SMAPE, CRPS) for scoring in parallel
- Supports **recursive and direct** forecast strategies

View the [full walkthrough](forecasting.md) on forecasting with `functime`.

## Embeddings

Temporal embeddings measure the relatedness of time-series.
Embeddings are more accurate and efficient compared to statistical methods (e.g. Catch22) for characteristing time-series.[^1]
Embeddings have applications across many domains from finance to IoT monitoring.
They are commonly used for the following tasks:

- **Matching:** Where time-series are ranked by similarity to a given time-series
- **Classification:** Where time-series are grouped together by matching patterns
- **Clustering:** Where time-series are assigned labels (e.g. normal vs irregular heart rate)
- **Anomaly detection:** Where outliers with unexpected regime / trend changes are identified

View the [full walkthrough](embeddings.md) on temporal embeddings with `functime`.

[^1]: Middlehurst, M., Schäfer, P., & Bagnall, A. (2023). Bake off redux: a review and experimental evaluation of recent time series classification algorithms. arXiv preprint arXiv:2304.13029.

## Quick Examples

??? info "Input Data Schemas"

    Forecasters, preprocessors, and splitters take a **panel dataset** where the first two columns represent entity (e.g. commodty name) and time (e.g. date). Subsequent columns represent observed values (e.g. price).

    ```
    >>> y_panel
    shape: (47_583, 3)

    commodity_type   time         price
    ------------------------------------
    Aluminum         1960-01-01    511.47
                     1960-02-01    511.47
                     1960-03-01    511.47
                     1960-04-01    511.47
                     1960-05-01    511.47
    ...                     ...       ...
    Zinc             2022-11-01   2938.92
                     2022-12-01   3129.48
                     2023-01-01   3309.81
                     2023-02-01   3133.84
                     2023-03-01   2967.46
    ```

    The `functime.embeddings.embed()` function takes a **wide dataset** where each row represents a single time-series.

    ```
    >>> X_y_wide
    shape: (150, 151)

    label     t0        t1     ...    t148      t149
    --------------------------------------------------
    1     -1.125013 -1.131338  ... -1.206178 -1.218422
    2     -0.626956 -0.625919  ... -0.612058 -0.606422
    2     -2.001163 -1.999575  ... -1.071147 -1.323383
    1     -1.004587 -0.999843  ... -1.044226 -1.043262
    1     -0.742625 -0.743770  ... -0.670519 -0.657403
    ...         ...       ...  ...       ...       ...
    2     -0.580006 -0.583332  ... -0.548831 -0.553552
    1     -0.728153 -0.730242  ... -0.686448 -0.690183
    2     -0.738012 -0.736301  ... -0.608616 -0.612177
    2     -1.265111 -1.256093  ... -1.193374 -1.192835
    1     -1.427205 -1.408303  ... -1.153119 -1.222043
    ```

### Forecasting

```python
import polars as pl
from functime.cross_validation import train_test_split
from functime.forecasting import lightgbm
from functime.metrics import mase

# Load commodities price data
y = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/commodities.parquet")
entity_col, time_col = y.columns[:2]

# Time series split
y_train, y_test = y.pipe(train_test_split(test_size=3))

# Fit-predict
model = lightgbm(freq="1mo", lags=24, max_horizons=3, strategy="ensemble")
model.fit(y=y_train)
y_pred = model.predict(fh=3)

# functime ❤️ functional design
# fit-predict in a single line
y_pred = lightgbm(freq="1mo", lags=24)(y=y_train, fh=3)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```
### Classification

Only available on `functime` Enterprise.

The following dataset represents velocity measurements from two gunslingers (label 1 and label 2) over 150 time periods (columns t0, t1, ..., t149) over 75 trials (rows).
In this example, we assign each sequence of measurement to one of the two gunsligners.

```python
import polars as pl
import functime
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Load GunPoint dataset (150 observations, 150 timestamps)
X_y_train = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/gunpoint_train.parquet")
X_y_test = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/gunpoint_test.parquet")

# Train-test split
X_train, y_train = (
    X_y_train.select(pl.all().exclude("label")),
    X_y_train.select("label")
)
X_test, y_test = (
    X_y_test.select(pl.all().exclude("label")),
    X_y_test.select("label")
)

X_train_embs = functime.embeddings.embed(X_train)

# Fit classifier on the embeddings
classifier = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
classifier.fit(X_train_embs, y_train)

# Predict and
X_test_embs = embed(X_test)
labels = classifier.predict(X_test_embs)
accuracy = accuracy_score(predictions, y_test)
```

### Clustering

Only available on `functime` Enterprise.

In this example, we cluster S&P 500 companies into groups with similar price patterns.

```python
import functime
import polars as pl
from hdbscan import HDBSCAN
from umap import UMAP
from functime.preprocessing import roll

# Load S&P500 panel data from 2022-06-01 to 2023-06-01
# Columns: ticker, time, price
y = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/sp500.parquet")

# Reduce noise by smoothing the time series using
# functime's `roll` function: 60-days moving average
y_ma_60 = (
    y.pipe(roll(window_sizes=[60], stats=["mean"], freq="1d"))
    .drop_nulls()
    # Pivot from panel to wide format
    .pivot(
        values="price__rolling_mean_60",
        columns="time",
        index="ticker"
    )
    # Remember all functime transforms are lazy!
    .collect()
)

# Create embeddings
embeddings = functime.embeddings.embed(y_ma_60)

# Reduce dimensionality with UMAP
reducer = UMAP(n_components=500, n_neighbors=10, metric="manhattan")
umap_embeddings = reducer.fit_transform(embeddings)

# Cluster with HDBSCAN
clusterer = HDBSCAN(metric="minkowski", p=1)
estimator.fit(X)

# Get predicted cluster labels
labels = estimator.predict(X)
```

### Splitters
View API reference for [`functime.cross_validation`](https://docs.functime.ai/ref/cross-validation/).
`functime` currently supports expanding window and rolling window splitters.
Splitters are used for cross-validation and backtesting.

### Preprocessing
View API reference for [`functime.preprocessing`](https://docs.functime.ai/ref/cross-validation/).
Preprocessors take in a `polars.DataFrame` or `polars.LazyFrame` as input and **always returns a `polars.LazyFrame`**.
No computation is run until the collect() method is called on the LazyFrame.
This allows Polars to [optimize the whole query](https://pola-rs.github.io/polars-book/user-guide/lazy/optimizations/) before execution.

```python
from functime.preprocessing import boxcox, impute

# Use df.pipe to chain operations together
X_new: pl.LazyFrame = (
    X.pipe(boxcox(method="mle"))
    .pipe(impute(method="linear"))
)
# Call .collect to execute query
```

## Time Series Data

### External Data

Easily enrich your own data with `functime`'s built-in datasets. Datasets include calendar effects, holidays, weather patterns, economic data, and seasonality features (i.e. Fourier Series).

### Example Data

It is easy to get started with `functime`.
Our GitHub repo contains a growing number of time-series data stored as `parquet` files:

- M4 Competition (daily, weekly, monthly, quarterly, yearly)[^2]
- M5 Competition[^3]
- Australian tourism[^4]
- Commodities prices[^5]
- User laptop activity[^6]
- Gunpoint measurements[^7]
- Japanese vowels [^8]

[^2]: https://mofc.unic.ac.cy/m4/
[^3]: https://mofc.unic.ac.cy/m5-competition/
[^4]: https://www.abs.gov.au/statistics/industry/tourism-and-transport/overseas-arrivals-and-departures-australia
[^5]: https://www.imf.org/en/Research/commodity-prices
[^6]: https://www.sciencedirect.com/science/article/pii/S2352340920306612
[^7]: http://www.timeseriesclassification.com/description.php?Dataset=GunPoint
[^8]: http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

## Contact Us

Book a quick 15 minute discovery call on [Calendly](https://calendly.com/functime-indexhub).
