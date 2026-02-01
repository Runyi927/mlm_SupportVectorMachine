import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class BaselineModel:
    """
    A simple baseline model that predicts the most frequent class.
    Follows the structure required by the project documentation.
    """
    def __init__(self):
        self.most_frequent_class = None

    def fit(self, X, y):
        # Find the most frequent class (e.g., Not a Superhost)
        self.most_frequent_class = y.value_counts().idxmax()

    def predict(self, X):
        # Predict the most frequent class for all samples
        return np.full(X.shape[0], self.most_frequent_class)

    def evaluate(self, X, y):
        # Compute metrics for Deliverable D2
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }