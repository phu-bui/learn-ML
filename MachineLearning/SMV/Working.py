import sklearn
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
scanner =   datasets.load_breast_cancer()

print(scanner.feature_names)
print(scanner.target_names)

x = scanner.data
y = scanner.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("ACC: ", acc)