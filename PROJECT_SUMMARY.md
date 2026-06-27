# KNN Iris Classification — Project Summary

## 1. Project Overview

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify an
Iris flower into one of three species:

- Setosa
- Versicolor
- Virginica

The model uses four flower measurements as input:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

This is a **supervised, multiclass classification** project because the model
learns from labeled examples and predicts one of three classes.

## 2. Dataset

The project uses scikit-learn's built-in Iris dataset:

- Total samples: 150
- Features: 4 numeric measurements
- Classes: 3
- Samples per class: 50
- Training samples: 120
- Test samples: 30

The data is split 80/20 using `random_state=42`. Stratification keeps the class
distribution balanced in both sets.

## 3. How KNN Works

KNN classifies a new sample using these steps:

1. Calculate its distance from all training samples.
2. Select the `k` nearest samples.
3. Count the classes of those neighbors.
4. Predict the class receiving the most votes.

The project selected **k = 5**, so the five nearest flowers vote on each
prediction.

A small `k` can overfit noise, while a large `k` can underfit by making the
decision boundary too smooth.

## 4. Why Feature Scaling Is Used

KNN is based on distance. A feature with a larger numerical range could
dominate the distance calculation.

`StandardScaler` standardizes every feature so they have comparable influence.
The scaler and KNN classifier are combined in a scikit-learn `Pipeline`. This:

- Applies identical preprocessing during training and prediction.
- Fits the scaler only on training data.
- Prevents test-data leakage during cross-validation.

## 5. Model Workflow

1. Load the Iris dataset.
2. Create a stratified 80/20 train-test split.
3. Build a pipeline containing `StandardScaler` and KNN.
4. Test odd values of `k` from 1 to 15.
5. Use five-fold cross-validation to select the best `k`.
6. Evaluate the selected model on untouched test data.
7. Save the complete pipeline and evaluation files.
8. Load the saved model to classify new flower measurements.

`GridSearchCV` chooses `k` systematically instead of relying on a guess.

## 6. Results

| Metric | Result |
|---|---:|
| Best k | 5 |
| Cross-validation accuracy | 96.7% |
| Test accuracy | 93.3% |
| Correct test predictions | 28 out of 30 |

### Confusion Matrix

| Actual / Predicted | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 10 | 0 |
| Virginica | 0 | 2 | 8 |

The model classified every Setosa and Versicolor sample correctly. Two
Virginica samples were classified as Versicolor because those species have
more overlapping measurements.

## 7. Important Evaluation Terms

- **Accuracy:** Percentage of all predictions that are correct.
- **Precision:** Of the samples predicted as a class, how many are correct?
- **Recall:** Of the actual samples in a class, how many were found?
- **F1-score:** Balance between precision and recall.
- **Confusion matrix:** Shows correct predictions and which classes were
  confused.

Accuracy is meaningful here because all three classes are balanced.

## 8. Project Files

| File | Purpose |
|---|---|
| `train_knn.py` | Trains, tunes, evaluates, and saves the model |
| `predict.py` | Uses the saved model for a new prediction |
| `test_knn.py` | Checks that model accuracy meets expectations |
| `pyproject.toml` | Lists project dependencies |
| `uv.lock` | Locks exact dependency versions |
| `README.md` | Provides setup and usage instructions |

## 9. How to Run

```powershell
uv sync
uv run python train_knn.py
uv run python -m unittest -v
uv run python predict.py 5.1 3.5 1.4 0.2
```

The sample input predicts **Setosa**.

## 10. Explanation for a Presentation

> I built a multiclass classification model using K-Nearest Neighbors to
> identify Iris flower species from four measurements. I used a stratified
> 80/20 split and combined StandardScaler with KNN in a pipeline because KNN
> depends on distance. I tested odd values of k from 1 to 15 using five-fold
> cross-validation, which selected k equal to 5. The final model achieved 93.3%
> accuracy on unseen data, correctly classifying 28 of 30 test samples. I also
> saved the complete pipeline and added a prediction script and automated test.

## 11. Common Interview Questions

### What type of machine learning problem is this?

It is supervised multiclass classification because the training data is
labeled and the output has three possible classes.

### Why did you choose KNN?

KNN is simple, interpretable, and effective for small numeric datasets where
similar samples usually belong to the same class.

### What does k mean?

It is the number of nearest training samples that vote on a prediction. This
project selected five neighbors.

### How was k selected?

`GridSearchCV` tested odd values from 1 to 15 using five-fold cross-validation
and selected the value with the highest average validation accuracy.

### Why is scaling important?

KNN calculates distances. Without scaling, a feature with a larger range could
incorrectly dominate the result.

### Why use a Pipeline?

It keeps scaling and classification together, applies the same preprocessing
every time, and prevents data leakage during cross-validation.

### What is data leakage?

Data leakage happens when information from test or validation data influences
training. It produces unrealistically high evaluation results.

### Why use a separate test set?

It estimates how well the final model performs on unseen data. The test set is
not used to select `k`.

### Why use cross-validation?

It evaluates every candidate across several training-validation splits, making
hyperparameter selection more reliable than using one validation split.

### What does 93.3% accuracy mean?

The model correctly predicted 28 of the 30 unseen test samples. It does not
guarantee exactly the same performance on all future data.

### What are KNN's limitations?

Prediction becomes slow with large datasets, it stores all training data,
irrelevant features affect distance, and it performs poorly in very high
dimensions.

### What is overfitting in KNN?

With a very small `k`, the model may memorize noise and perform well on
training data but poorly on new data.

### How would you improve the project?

Possible improvements include:

- Compare Euclidean and Manhattan distance.
- Test distance-weighted voting.
- Add stricter input validation.
- Visualize the dataset and model errors.
- Build a Streamlit interface or FastAPI endpoint.
- Monitor input quality and model accuracy after deployment.

## 12. Final Points to Remember

- KNN predicts using the classes of nearby training samples.
- Scaling is essential because KNN is distance-based.
- The pipeline prevents preprocessing mistakes and leakage.
- Cross-validation selected `k = 5`.
- Test accuracy is 93.3%, with 28 correct predictions out of 30.
- The complete fitted pipeline is saved for reusable predictions.

