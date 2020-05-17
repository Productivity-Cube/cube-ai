import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#get data
df = quandl.get("WIKI/GOOGL", authtoken="-sdub9xfGumA6h-9pcC7")

#prepare labels
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# print(df.head())
#prepare data
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

# print(df.head())
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
X = X[:-forecat_out]
X_lately = X[-forecast_out:]
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

#testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
# clf = svm.SVR() #differen algorithm
clf.fit(X_train, y_train)

#get our confidence accuracy
accuracy = clf.score(X_test, y_test)
print(accuracy)


