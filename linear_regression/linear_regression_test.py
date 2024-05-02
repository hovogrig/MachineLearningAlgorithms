from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

diabetes = load_diabetes()
xTrain, xTest, yTrain, yTest = train_test_split(diabetes['data'], diabetes['target'], test_size=0.2, random_state=10)

print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

yTest = yTest.reshape(-1, 1)
yTrain = yTrain.reshape(-1, 1)


lin_reg = LinearRegression()
lin_reg.fit(xTrain, yTrain)

y_pred = lin_reg.predict(xTest)


print("R^2 score:", lin_reg.r2_score(yTest, y_pred))
