# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:05:12 2020

@author: spand
"""
# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import LinearSVC

dataset = pd.read_csv('HiveFinal.csv')
dataset = dataset.fillna(0)
column_names = dataset.columns[4:]

np_arr = dataset.iloc[:, 4:]
#print(type(np_arr))
data = pd.DataFrame(np_arr, columns = column_names)
X = dataset.iloc[:, 4:23].values
y = dataset.iloc[:, -1].values



from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

clf_smote = LinearSVC().fit(X_resampled, y_resampled)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.4, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#ROC_AUC 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.3f' % recall)


from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.3f' % precision)

from sklearn.metrics import f1_score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)
