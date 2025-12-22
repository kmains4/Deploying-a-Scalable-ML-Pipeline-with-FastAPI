import numpy as np
import pytest

from ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def sample_data():
    """
    Create simple numeric sample data for testing.
    """
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    categorical_features = []
    return X, y, categorical_features


def test_train_model(sample_data):
    """
    Test that the model trains and returns a model object.
    """
    X, y, categorical_features = sample_data
    model = train_model(X, y, categorical_features)
    assert model is not None


def test_inference(sample_data):
    """
    Test that inference returns predictions of correct length.
    """
    X, y, categorical_features = sample_data
    model = train_model(X, y, categorical_features)
    preds = inference(model, X)
    assert len(preds) == len(y)


def test_compute_model_metrics():
    """
    Test that model metrics return non-negative values.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision >= 0
    assert recall >= 0
    assert fbeta >= 0
