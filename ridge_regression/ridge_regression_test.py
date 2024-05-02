import numpy as np
from sklearn.model_selection import train_test_split
from ridge_regression import RidgeRegression


# Generate some sample data
np.random.seed(10)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features)
y = X.dot(true_coef) + np.random.normal(0, 0.5, size=n_samples) # Adding some noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Train ridge regression model
ridge = RidgeRegression(alpha=0.01)  # You can adjust the alpha (λ) parameter here
ridge.fit(X_train, y_train)


# Make predictions
y_pred = ridge.predict(X_test)

# Evaluate performance
r2 = ridge.r2_score(y_test, y_pred)

print("R^2 Score:", r2)
