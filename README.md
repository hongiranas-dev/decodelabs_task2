# KNN Iris Classifier

This project trains a K-Nearest Neighbors (KNN) model to classify Iris flowers
as **setosa**, **versicolor**, or **virginica** from four numeric measurements.

The training pipeline:

1. Loads scikit-learn's built-in Iris dataset.
2. Creates a stratified 80/20 train/test split.
3. Standardizes the features so distance calculations are meaningful.
4. Uses 5-fold cross-validation to choose the best odd `k` from 1 through 15.
5. Evaluates the selected model and saves it for later predictions.

## Run

Install dependencies and train:

```powershell
uv sync
uv run python train_knn.py
```

Classify a flower (measurements are in centimeters):

```powershell
uv run python predict.py 5.1 3.5 1.4 0.2
```

Run the test:

```powershell
uv run python -m unittest -v
```

## Outputs

Training writes these files to `outputs/`:

- `knn_iris_model.joblib`: fitted scaling and KNN pipeline plus class metadata
- `metrics.json`: accuracy, selected `k`, and per-class scores
- `confusion_matrix.csv`: actual-versus-predicted class counts
- `test_predictions.csv`: held-out samples and their predictions
