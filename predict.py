"""Classify one Iris flower using a saved KNN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sepal_length", type=float, help="Sepal length in cm")
    parser.add_argument("sepal_width", type=float, help="Sepal width in cm")
    parser.add_argument("petal_length", type=float, help="Petal length in cm")
    parser.add_argument("petal_width", type=float, help="Petal width in cm")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/knn_iris_model.joblib"),
        help="Path to the trained model artifact.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    features = [
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,
    ]
    class_index = int(artifact["model"].predict([features])[0])
    probabilities = artifact["model"].predict_proba([features])[0]

    print(f"Predicted class: {artifact['target_names'][class_index]}")
    print("Class probabilities:")
    for name, probability in zip(artifact["target_names"], probabilities):
        print(f"  {name}: {probability:.1%}")


if __name__ == "__main__":
    main()

