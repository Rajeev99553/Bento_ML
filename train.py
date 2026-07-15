import bentoml

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X,y = iris.data, iris.target

clf = svm.SVC(gamma = "scale")
clf.fit(X,y)


# Model saved in c-drive--->Users--->BentoML--->Models
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")

## iris_clf:d5ijo2b6hgo3i2b6

