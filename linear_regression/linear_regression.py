import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        self.intercept_ = 0
        self.coef_ = np.zeros((X.shape[1], 1))
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            errors = y_pred - y
            d_intercept = (2 / X.shape[0]) * np.sum(errors)
            d_coef = (2 / X.shape[0]) * X.T.dot(errors)
            self.intercept_ = self.intercept_ - self.learning_rate * d_intercept
            self.coef_ = self.coef_ - self.learning_rate * d_coef

    def predict(self, X):
        y_pred = self.intercept_ + X.dot(self.coef_)
        return y_pred

    def r2_score(self, y_true, y_pred):
        mean = np.mean(y_true)
        ssr = np.sum((y_true - y_pred)**2)
        sst = np.sum((y_true - mean)**2)
        return 1 - (ssr / sst)
