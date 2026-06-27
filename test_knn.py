"""Basic tests for the KNN training pipeline."""

import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from train_knn import RANDOM_STATE, build_model


class KnnModelTests(unittest.TestCase):
    def test_model_reaches_expected_accuracy(self) -> None:
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data,
            iris.target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=iris.target,
        )
        model = build_model()
        model.fit(x_train, y_train)

        self.assertGreaterEqual(model.score(x_test, y_test), 0.9)
        self.assertIn(model.best_params_["knn__n_neighbors"], range(1, 16, 2))


if __name__ == "__main__":
    unittest.main()

