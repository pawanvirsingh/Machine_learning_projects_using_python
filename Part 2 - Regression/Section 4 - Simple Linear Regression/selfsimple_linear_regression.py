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
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#data preprocessing step this is 

from sklearn.cross_validation  import train_test_split
X_train,x_test, y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)


#data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_predict = regressor.predict(X_test)
