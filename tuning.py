# using 10-Fold Cross Validation on your training data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn import cross_validation as cval
cval.cross_val_score(model, X_train, y_train, cv=10)
cval.cross_val_score(model, X_train, y_train, cv=10).mean()


#power tuning
from sklearn import svm, grid_search, datasets

iris = datasets.load_iris()
model = svm.SVC()

#using gridsearch
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
classifier = grid_search.GridSearchCV(model, parameters)
classifier.fit(iris.data, iris.target)

#using RandomizedSearchCV
parameter_dist = {
  'C': scipy.stats.expon(scale=100),
  'kernel': ['linear'],
  'gamma': scipy.stats.expon(scale=.1),
}
classifier = grid_search.RandomizedSearchCV(model, parameter_dist)
classifier.fit(iris.data, iris.target)
