from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from knn_classifier import KnnClassifier

cancer_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, random_state=66)

classifier = KnnClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
print("Accuracy: ", classifier.score(X_test, y_test))
