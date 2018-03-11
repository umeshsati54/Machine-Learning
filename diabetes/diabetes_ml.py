import numpy as np
import pandas as pd

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 8].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer.fit(X)
X = imputer.transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
classifier = RandomForestClassifier(n_estimators=130, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(94+34)/(154)