from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

X_wine, y_wine = load_wine(return_X_y=True)

X_wine = X_wine[(y_wine == 0) | (y_wine == 1)]
y_wine = y_wine[(y_wine == 0) | (y_wine == 1)]

X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, random_state=42)
y_wine_train = y_wine_train.reshape(-1, 1) 
y_wine_test = y_wine_test.reshape(-1, 1)

print("Shapes:", X_wine_train.shape, X_wine_test.shape, y_wine_train.shape, y_wine_test.shape)

log_reg = LogisticRegression()

log_reg.fit(X_wine_train, y_wine_train)

y_wine_predict_test = log_reg.predict(X_wine_test)

print("Confusion matrix:")
print(log_reg.confusion_matrix(y_wine_test, y_wine_predict_test))

print("f1-score:", log_reg.f1_score(y_wine_test, y_wine_predict_test))

