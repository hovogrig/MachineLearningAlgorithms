import numpy as np


class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.intercept = None
        self.coef = None

    def fit(self, X, y):
        self.intercept = 0
        self.coef = np.zeros((X.shape[1], 1))
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            errors = y_pred - y
            d_intercept = (2 / X.shape[0]) * np.sum(errors)
            d_coef = (2 / X.shape[0]) * X.T.dot(errors) + 2 * self.alpha * self.coef
            self.intercept = self.intercept - self.learning_rate * d_intercept
            self.coef = self.coef - self.learning_rate * d_coef

    def predict(self, X):
        y_pred = self.intercept + X.dot(self.coef)
        return y_pred

    def r2_score(self, y_true, y_pred):
        mean = np.mean(y_true)
        ssr = np.sum((y_true - y_pred)**2)
        sst = np.sum((y_true - mean)**2)
        return 1 - (ssr / sst)

