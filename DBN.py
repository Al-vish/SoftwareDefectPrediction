# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:09:04 2020

@author: Vishal
"""

# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import LinearSVC
from dbn.tensorflow import SupervisedDBNClassification


dataset = pd.read_csv('GeodFeinal.csv')
dataset = dataset.fillna(0)
column_names = dataset.columns[4:]

np_arr = dataset.iloc[:, 4:]
#print(type(np_arr))
data = pd.DataFrame(np_arr, columns = column_names)
X = dataset.iloc[:, 4:23].values
y = dataset.iloc[:, -1].values

from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = ADASYN().fit_resample(X, y)

clf_smote = LinearSVC().fit(X_resampled, y_resampled)

X_resampled = X_resampled.astype(np.float32)


############### feature scaling
from sklearn.preprocessing import StandardScaler
X_resampled= StandardScaler().fit_transform(X_resampled)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2, random_state = 0)



classifier = SupervisedDBNClassification(hidden_layers_structure =       [256, 256],
learning_rate_rbm=0.05,
learning_rate=0.1,
n_epochs_rbm=10,
n_iter_backprop=100,
batch_size=32,
activation_function='relu',
dropout_p=0.2)

classifier.fit(X_train, y_train)









