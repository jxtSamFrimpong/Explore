import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, datasets

cancer = datasets.load_breast_cancer()

print('Features', cancer.feature_names)
print('Targets', cancer.target_names)

x = cancer.data
y = cancer.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

prediction_classes = cancer.target_names

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)