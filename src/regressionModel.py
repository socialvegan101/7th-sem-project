import numpy as np


# # --- Custom Linear Regression (From Scratch) ---
class CustomLinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.n_features = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features = n_features

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)

        # FIX: ensure correct shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # safety check (prevents your exact error)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature mismatch! Model expects {self.n_features}, "
                f"but got {X.shape[1]}"
            )
        return np.dot(X, self.weights) + self.bias