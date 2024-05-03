import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.0001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.intercept = None

    def fit(self, X_train, y_train):
        self.weights = np.zeros((X_train.shape[1], 1))
        self.intercept = 0

        for _ in range(self.n_iterations):

            y_pred = self.predict_proba(X_train)
            errors = y_pred - y_train

            self.intercept = self.intercept - self.learning_rate * ((1 / X_train.shape[0]) * np.sum(errors))
            self.weights = self.weights - self.learning_rate * ((1 / X_train.shape[0]) * (X_train.T.dot(errors)))

    def predict(self, X, threshold=0.5):
        predictions = self.__sigmoid(self.intercept + X.dot(self.weights))
        binary_predictions = (predictions > threshold).astype(int)
        return binary_predictions

    def predict_proba(self, X): 
        return self.__sigmoid(self.intercept + X.dot(self.weights))

    def confusion_matrix(self, y_true, y_pred):
        true_negatives, false_positives, false_negatives, true_positives = 0, 0, 0, 0 
        for true, pred in zip(y_true, y_pred):
            if true == 0:
                if pred == 0:
                    true_negatives += 1
                else:
                    false_positives += 1
            else:
                if pred == 0:
                    false_negatives += 1
                else:
                    true_positives += 1
        return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

    def accuracy_score(self, y_true, y_pred):
        accuracy = (y_true == y_pred).sum() / len(y_true)
        return accuracy

    def f1_score(self, y_true, y_pred):
        confusion_matrix = self.confusion_matrix(y_true, y_pred)
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        return (2*tp) / (2*tp+(fp+fn))        

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
