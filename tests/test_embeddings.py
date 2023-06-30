import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functime.embeddings import Embedder

DATA_URL = "https://github.com/descendant-ai/functime/raw/main/data/"


@pytest.fixture(scope="module")
def gunpoint_dataset():
    """Equal length, univariate time series."""
    X_y_train = pl.read_parquet(f"{DATA_URL}/gunpoint_train.parquet")
    X_y_test = pl.read_parquet(f"{DATA_URL}/gunpoint_test.parquet")
    X_train = X_y_train.select(pl.all().exclude("label"))
    y_train = X_y_train.select("label")
    X_test = X_y_test.select(pl.all().exclude("label"))
    y_test = X_y_test.select("label")
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def basicmotions_dataset():
    try:
        from aeon.datasets import load_basic_motions
    except ImportError as e:
        raise ImportError(
            "Please install aeon to run this test: pip install aeon"
        ) from e
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    return X_train, X_test, y_train, y_test


def test_minirocket_on_gunpoint(gunpoint_dataset):
    # each row is 1 time series, each column is 1 time point
    X_training, X_test, Y_training, Y_test = gunpoint_dataset

    # Keep everything as np.ndarray
    X_training = X_training
    X_test = X_test
    Y_training = Y_training.to_numpy().ravel()
    Y_test = Y_test.to_numpy().ravel()

    emb = Embedder(model="minirocket")
    # Minirocket takes in numpy array with columnar format
    X_training_transform = emb.fit_transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))
    assert emb.is_fitted
    assert emb.is_multivariate is False
    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # transform test data
    X_test_transform = emb.transform(X_test, model="minirocket")

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 9_996))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)
    print(f"Accuracy: {accuracy}")

    # test predictions (on Gunpoint, should be > 99% accurate)
    assert accuracy > 0.97


@pytest.mark.parametrize("size", [25, 50, 75, 100, 125, 150])
@pytest.mark.benchmark(group="embeddings")
def test_benchmark_minirocket(benchmark, gunpoint_dataset, size):
    # each row is 1 time series, each column is 1 time point
    X_training, *_ = gunpoint_dataset

    # Keep everything as np.ndarray
    X_training = X_training.to_numpy()[:size]

    # Minirocket takes in numpy array with columnar format
    X_training_transform = benchmark(embed, X_training, "minirocket")

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))


def test_manyrocket(basicmotions_dataset):
    X_training, X_test, Y_training, Y_test = basicmotions_dataset

    print(X_training.shape)
    print(X_test.shape)
    print(Y_training.shape)
    print(Y_test.shape)
    # 'fit' MINIROCKET -> infer data dimensions, generate random kernels
    # Rustyrocket takes in numpy array with columnar format

    # transform training data
    X_training_transform: np.ndarray = embed(X_training, model="manyrocket")

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # transform test data
    X_test_transform = embed(X_test, model="manyrocket")

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 9_996))

    # predict (alternatively:X_training 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on BasicMotions, should be 100% accurate)
    assert accuracy == 1.0


@pytest.mark.parametrize("size", [25, 50, 75, 100, 125, 150])
@pytest.mark.benchmark(group="embeddings")
def test_benchmark_manyrocket(benchmark, basicmotions_dataset, size):
    # Dims: (instance, feature/dimension, time)
    X_training, *_ = basicmotions_dataset

    # Keep everything as np.ndarray
    X_training = X_training.to_numpy()[:size]

    # Minirocket takes in numpy array with columnar format
    X_training_transform = benchmark(embed, X_training, "manyrocket")

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))
