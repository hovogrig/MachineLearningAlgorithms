# Machine Learning Algorithms Implementation

This repository contains Python implementations of various machine learning algorithms from scratch. These implementations are meant for educational purposes to understand the underlying principles of these algorithms.

## Algorithms Implemented

1. K-Nearest Neighbors (KNN)
2. Linear Regression
3. Logistic Regression
4. Ridge Regression
5. PCA

## Requirements

- Python 3.x
- NumPy
- scikit-learn
- seaborn
- matplotlib

## Installation

You can install the required packages using pip3 with the following command:

```bash
pip3 install -r requirements.txt
```

## Usage

Each algorithm is implemented in its own Python file. You can find the implementations in the following files:

- `knn_classifier.py`: K-Nearest Neighbors
- `linear_regression.py`: Linear Regression
- `logistic_regression.py`: Logistic Regression
- `ridge_regression.py`: Ridge Regression

To use any of these algorithms, you can import the respective Python file into your project or use them directly in a Python script. 

Here's a simple example of how to use the Linear Regression implementation:

```python
from linear_regression import LinearRegression

# Example data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict([[6]])
print(predictions)  # Output: [12.0]
