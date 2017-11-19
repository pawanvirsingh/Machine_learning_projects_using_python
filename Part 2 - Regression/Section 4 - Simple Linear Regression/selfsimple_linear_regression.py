#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:03:36 2017

@author: pawan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset= pd.read_csv("Salary_Data.csv")
X =dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#data prep rocessing step this is 

from sklearn.cross_validation  import train_test_split
X_train,x_test, y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)


#data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_predict = regressor.predict(x_test)

"""
error = y_test-y_predict
e= np.sum(error)
abserror = np.absolute(error)
sqrarr= np.square(error)
addall= np.sum(sqrarr)
"""
#for ploting the data  for train data
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary Vs Experience (train)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#for ploting the data  for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title("Salary Vs Experience (Test data set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()