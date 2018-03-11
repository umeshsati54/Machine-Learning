 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('battles.csv')
test_dataset = pd.read_csv('battles_g_sheets.csv')

test_it = test_dataset.iloc[:,:].values

X = dataset.iloc[:, [1, 3, 4, 5, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].values
dataset.isnull().any()


# mean in 17 and 18##
#one hot  3 4 5 9 14 19 20 22 23
#missing = 17 18
y = dataset.iloc[:, [13]].values

from sklearn.preprocessing import Imputer
imputer_X_mean = Imputer(missing_values='NaN', strategy='mean', axis=0 )
imputer_X_mean = imputer_X_mean.fit(X[:, [8, 9]])
X[:, [8, 9]] = imputer_X_mean.transform(X[:, [8, 9]])
'''imputer_X = Imputer(missing_values='NaN', strategy='most_frequent', axis=0 )
imputer_X = imputer_X.fit(X[:, [0,14]])
X[:, [0,13]] = imputer_X.transform(X[:, [0,13]])
# [0,1,2,3,4,5,6,7,10,11,12,13,14]'''


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_Y = LabelEncoder()
y[:-1,0] = labelencoder_Y.fit_transform(y[:-1, 0])
#[1,2,3,4,5,10,11,13,14]
labelencoder_X = LabelEncoder() 
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,11] = labelencoder_X.fit_transform(X[:,11])
X[:,13] = labelencoder_X.fit_transform(X[:,13])
X[:,14] = labelencoder_X.fit_transform(X[:,14])

test_it[:, 1] = labelencoder_X.fit_transform(test_it[:, 1])
test_it[:, 2] = labelencoder_X.fit_transform(test_it[:, 2])
test_it[:, 3] = labelencoder_X.fit_transform(test_it[:, 3])
test_it[:, 4] = labelencoder_X.fit_transform(test_it[:, 4])
test_it[:, 5] = labelencoder_X.fit_transform(test_it[:, 5])
test_it[:, 10] = labelencoder_X.fit_transform(test_it[:, 10])
test_it[:, 11] = labelencoder_X.fit_transform(test_it[:, 11])
test_it[:, 13] = labelencoder_X.fit_transform(test_it[:, 13])
test_it[:, 14] = labelencoder_X.fit_transform(test_it[:, 14])

onehotencoder = OneHotEncoder(categorical_features=[1,2,3,4,5,10,11,13,14])
X = onehotencoder.fit_transform(X).toarray()
test_it = onehotencoder.fit_transform(test_it).toarray()


imputer_y = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer_y = imputer_y.fit(y)
y = imputer_y.transform(y)





# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(test_dataset)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







