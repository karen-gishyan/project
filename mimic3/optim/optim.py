from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from utils import remove_nan
from models import MultiClassLogisticRegression
import warnings

warnings.filterwarnings("ignore")

heart_disease = fetch_ucirepo(id=45)
print("heart_disease dataset shape {}".format(heart_disease.data.features.shape))
X = heart_disease.data.features
y = heart_disease.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr=MultiClassLogisticRegression()
lr.fit(X_train,y_train)
print("Custom SGD",round(lr.score(X_test,y_test),2))
print("Predicted test labels",lr.predict_classes(X_test))
# lr.plot()

clf=SGDClassifier(loss='log_loss',random_state=0)
clf.fit(X_train,y_train)
print("SGD",round(clf.score(X_test,y_test),2))
print("Predicted test labels",clf.predict(X_test))

####
hepatitis = fetch_ucirepo(id=46)
print("hepatitis dataset shape {}".format(hepatitis.data.features.shape))
X = hepatitis.data.features
y = hepatitis.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr=MultiClassLogisticRegression()
lr.fit(X_train,y_train)
print("Custom SGD",round(lr.score(X_test,y_test),2))
print("Predicted test labels",lr.predict_classes(X_test))
# lr.plot()

clf=SGDClassifier(loss='log_loss',random_state=0)
clf.fit(X_train,y_train)
print("SGD",round(clf.score(X_test,y_test),2))
print("Predicted test labels",clf.predict(X_test))

# ####
lung_cancer = fetch_ucirepo(id=62)
print("lung_cancer dataset shape {}".format(lung_cancer.data.features.shape))
X = lung_cancer.data.features
y = lung_cancer.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr=MultiClassLogisticRegression()
lr.fit(X_train,y_train)
print("Custom SGD",round(lr.score(X_test,y_test),2))
print("Predicted test labels",lr.predict_classes(X_test))
# lr.plot()

clf=SGDClassifier(loss='log_loss',random_state=0)
clf.fit(X_train,y_train)
print("SGD",round(clf.score(X_test,y_test),2))
print("Predicted test labels",clf.predict(X_test))