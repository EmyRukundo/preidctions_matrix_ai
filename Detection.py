# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd;
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


##load the dataset
data = pd.read_csv("pima-indians-diabetes.csv");

#split the trainning set and the test set
X = data.iloc[:, 0:8].values
y = data.iloc[:,8].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)


#building know the nn model
model = Sequential()
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


#compile the model
model.compile(optimizer="adam", loss =  "binary_crossentropy", metrics="accuracy")

#fitting the model
model.fit(X_train, y_train, epochs = 100, batch_size = 10)

#testing the model
predictions = model.predict(X_test)
predictions = (predictions > 0.5)

comparison = confusion_matrix(y_test, predictions)




















