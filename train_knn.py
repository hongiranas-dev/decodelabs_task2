"""Train and evaluate a K-Nearest Neighbors classifier on the Iris dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def build_model() -> GridSearchCV:
    """Create a scaled KNN pipeline and tune the number of neighbors."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )
    return GridSearchCV(
        pipeline,
        {"knn__n_neighbors": list(range(1, 16, 2))},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )


def train(output_dir: Path) -> dict:
    """Train the classifier, save artifacts, and return evaluation metrics."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    search = build_model()
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "dataset": "Iris",
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "best_k": int(search.best_params_["knn__n_neighbors"]),
        "cross_validation_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": search.best_estimator_,
            "feature_names": list(iris.feature_names),
            "target_names": list(iris.target_names),
        },
        output_dir / "knn_iris_model.joblib",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test, predictions)
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *iris.target_names])
        for label, row in zip(iris.target_names, matrix):
            writer.writerow([label, *row])

    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([*iris.feature_names, "actual", "predicted"])
        for features, actual, predicted in zip(x_test, y_test, predictions):
            writer.writerow(
                [
                    *features,
                    iris.target_names[actual],
                    iris.target_names[predicted],
                ]
            )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for the trained model and evaluation files.",
    )
    args = parser.parse_args()
    metrics = train(args.output_dir)
    print(f"Best k: {metrics['best_k']}")
    print(f"CV accuracy: {metrics['cross_validation_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

