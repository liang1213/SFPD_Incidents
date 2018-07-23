#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

train_data = pd.read_csv("../input/Police_Department_Incidents.csv")

target = train_data["Category"].unique()
print(target.shape)
target

X = train_data.drop(train_data.columns[[0, 2, 7, 8, -2, -1]], axis = 1)

y = train_data.iloc[:, 1]


def preprocess_data(dataset):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Year'] = dataset.Date.apply(lambda x: x.year)
    dataset['Month'] = dataset.Date.apply(lambda x: x.month)
    dataset['Day'] = dataset.Date.apply(lambda x: x.day)
    dataset['Time'] = pd.to_datetime(dataset['Time'])
    dataset['Hour'] = dataset.Time.apply(lambda x: x.hour)
    dataset['Minute'] = dataset.Time.apply(lambda x: x.minute)
    dataset = dataset.drop(['Date', 'Time'],  1)
    
    return dataset

X = preprocess_data(X)

crime_year = X.groupby('Year')['Category'].count()

fig = plt.figure(figsize=(10, 6))

plt.plot(crime_year.index[:-1], crime_year.values[:-1], '-bo')

#plt.xticks(crime_year.index[:-1])

plt.tick_params(labelsize=14)

fig.suptitle('Number of Crimes in San Francisco Each Year 2003-2017', fontsize=18)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Crimes', fontsize=16)
fig.savefig('NumberCrimes_Year.png', dpi = 300)



crime_day = X[(X['Category'] == 'ROBBERY') & (X['PdDistrict'] == 'PARK') & (X['Year'] == 2017) ].groupby('DayOfWeek')['Category'].count()

dictionary = dict(zip(crime_day.keys(), crime_day.values))

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday', 'Sunday']

nums = [dictionary[i] for i in days]



fig = plt.figure(figsize=(10, 6))

sns.barplot(days, nums)
plt.tick_params(labelsize=14)

fig.suptitle('Number of "ROBBERY" in the "PARK" District 2017', fontsize=18)

plt.xlabel('Day of Week', fontsize=16)
plt.ylabel('Number of Crimes', fontsize=16)
fig.savefig('NumberCrimes_Day.png', dpi = 300)

X = X.drop('Year', axis = 1)

X = X.drop('Category', axis = 1)

X = pd.get_dummies(X, columns=['DayOfWeek', 'PdDistrict'])


from sklearn.preprocessing import LabelEncoder
y = y.to_frame()
le = LabelEncoder()
y["Category"] = le.fit_transform(y["Category"])

keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 10, n_estimators = 256)
rf.fit(X_train.values, y_train.values.ravel())


y_pred = rf.predict(X_test)
y_pred.shape
y_test.shape

from sklearn.metrics import accuracy_score
print ("Train Accuracy: ", accuracy_score(y_train, rf.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))


#('Train Accuracy: ', 0.2565971358094919)
#('Test Accuracy: ', 0.25541923906050723)
