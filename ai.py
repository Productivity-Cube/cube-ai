import pandas as pd
import quandl
import math
import urllib2
import csv
import sys
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from decouple import config
from numpy import genfromtxt
from urllib2 import urlopen
import os

#get data
BACKEND_URL = config('BACKEND_URL')
USER = sys.argv[1]
FILENAME = "storage\{}.csv".format(USER)
URL = "{}/api/user/{}/events.csv".format(BACKEND_URL, USER)

try:
    response = urlopen(URL)
except urllib2.HTTPError, e:
    print('ERROR')
else:
    html = response.read()
    with open(FILENAME, "w+") as text_file:
        text_file.write(html)

    data = np.genfromtxt(FILENAME, delimiter=",", skip_header=1)
    X = np.delete(data, 0, axis=1)
    y = data[:,0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

    clf = LinearRegression()
    clf.fit(X_train, y_train)

    #get our confidence accuracy
    accuracy = clf.score(X_test, y_test)

    #predict
    forecast_out = 1
    X_lately = X[:forecast_out]
    forecast_set  = clf.predict(X_lately)
    print(forecast_set[0], accuracy)
    os.remove(FILENAME)
