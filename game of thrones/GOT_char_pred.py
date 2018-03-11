import numpy as np
import pandas as pd

dataset = pd.read_csv('c_p_opt.csv')
dataset[['title']] = dataset[['title']].replace(np.NaN, 'nd' )
dataset[['house']] = dataset[['house']].replace(np.NaN, 'n' )
X = dataset.iloc[:, 1:15].values
y = dataset.iloc[:, 15:16].values
lm(dataset)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder() 
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
X[:,2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[0,2])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1946,1)).astype(int), values = X, axis = 1)
X_opt = X[:,614:625]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[614,615,616,617,618,620,621,622,623,624]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[614,615,616,617,618,620,622,623,624]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[614,615,616,617,618,620,622,623]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

