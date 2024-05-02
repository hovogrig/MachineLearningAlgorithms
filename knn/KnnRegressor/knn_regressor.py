import numpy as np


class KnnRegressor:
    def __init__(self, n_neighbors=5, metric='l2'):
        self.y_train = None
        self.x_train = None
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predicted_values = [self._predict(obs) for obs in x_test]
        return predicted_values

    def r2_score(self, x_test, y_test):
        predicted_values = self.predict(x_test)
        # calculate R^2
        mean = np.mean(y_test)
        ssr = sum([(predicted_value - actual_value)**2 for predicted_value, actual_value in zip(predicted_values, y_test)])
        sst = sum([(actual_value - mean)**2 for actual_value in y_test])
        return 1 - (ssr / sst)

    def _predict(self, obs):
        distances = np.array([self._distance(obs, train) for train in self.x_train])
        neighbors_indices = np.argsort(distances)[:self.n_neighbors]
        y_values = [self.y_train[neighbors_index] for neighbors_index in neighbors_indices]
        return np.mean(y_values)

    def _distance(self, point_1, point_2):
        p_distance = int(self.metric[1])
        sum_of_coordinates = sum(((x_1 - x_2) ** p_distance) for x_1, x_2 in zip(point_1, point_2))
        return sum_of_coordinates ** 1 / p_distance
