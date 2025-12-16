import joblib
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def train_model(X, y, categorical_features):
    """
    Trains a machine learning model and returns it.
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="passthrough"
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    clf.fit(X, y)
    return clf


def inference(model, X):
    """
    Run model inference and return predictions.
    """
    return model.predict(X)


def compute_model_metrics(y, preds):
    """
    Computes precision, recall, and fbeta.
    """
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1)

    return precision, recall, fbeta


def save_model(model, encoder, lb):
    """
    Save model and encoders.
    """
    joblib.dump(model, "model.joblib")
    joblib.dump(encoder, "encoder.joblib")
    joblib.dump(lb, "lb.joblib")


def load_model():
    """
    Load model and encoders.
    """
    model = joblib.load("model.joblib")
    encoder = joblib.load("encoder.joblib")
    lb = joblib.load("lb.joblib")
    return model, encoder, lb


def performance_on_categorical_slice(
    model,
    X,
    y,
    categorical_feature
):
    """
    Computes model metrics for each category in a feature slice.
    """
    results = []

    for category in X[categorical_feature].unique():
        idx = X[categorical_feature] == category
        X_slice = X[idx]
        y_slice = y[idx]

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        results.append(
            f"Slice: {categorical_feature} = {category}\n"
            f"Precision: {precision:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"F1: {fbeta:.3f}\n"
        )

    return results
