import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from ml.model import (
    train_model,
    save_model,
    performance_on_categorical_slice,
)

DATA_PATH = "data/census.csv"


def main():
    # Load data
    data = pd.read_csv(DATA_PATH)

    # Target and features
    y = data["salary"]
    X = data.drop("salary", axis=1)

    # Categorical features (required for rubric)
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Encode target
    lb = LabelBinarizer()
    y = lb.fit_transform(y).ravel()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train, categorical_features)

    # Save model and encoders (encoder is inside pipeline, pass None)
    save_model(model, None, lb)

    # Slice performance on ONE categorical feature (rubric requirement)
    slice_results = performance_on_categorical_slice(
        model,
        X_test,
        y_test,
        categorical_feature="sex",
    )

    # Write slice output to file
    with open("slice_output.txt", "w") as f:
        for result in slice_results:
            f.write(result + "\n")


if __name__ == "__main__":
    main()
