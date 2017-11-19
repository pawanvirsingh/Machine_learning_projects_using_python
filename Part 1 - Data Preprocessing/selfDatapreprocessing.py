#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:43:38 2017

@author: pawan
"""

import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
dataSet = pd.read_csv("Data.csv")
print(dataSet)
X= dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy="median",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#now train text split functionality 
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#for feature scaling of data to put alll data on same scale because the data should 
#same scale 
from sklearn.preprocessing import StandardScaler
stander=StandardScaler()
X_train=stander.fit_transform(X_train)
X_test= stander.transform(X_test)
