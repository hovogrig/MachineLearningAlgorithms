from sklearn.model_selection import train_test_split
import mglearn
from knn_regressor import KnnRegressor

# Create the regression dataset
X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KnnRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("Accuracy: ", reg.r2_score(X_test, y_test))
