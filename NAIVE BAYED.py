# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 00:32:24 2020

@author: Vishal
"""
# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import linear_model

# Import PySwarms
import pyswarms as ps

dataset = pd.read_csv('GeodeFinal.csv')
dataset = dataset.fillna(0)
X = dataset.iloc[:, 4:14].values
y = dataset.iloc[:, -1].values





"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1: 16])
X[:,1: 16] = imputer.transform(X[:,1: 16])"""

 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

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

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


#####################################################################
# Importing the dataset
dataset = pd.read_csv('BeamFinal.csv')

column_names = dataset.columns[4:]

np_arr = dataset.iloc[:, 4:]
print(type(np_arr))
data = pd.DataFrame(np_arr, columns = column_names)
X = dataset.iloc[:, 4:13].values
y = dataset.iloc[:, -1].values


class1 = []
for i in range(len(y)):
    if y[i] == 1:
        class1.append(i)

class_1 = np.array(class1)

class0 = []
for i in range(len(y)):
    if y[i] == 0:
        class0.append(i)

class_0 = np.array(class0)

 
nc_0 = len(class_0)
nc_1 = len(class_1)


diff = nc_0 - nc_1 


for i in range(diff):

    index = np.random.choice(class_1)
    x = data.iloc[index]
    # concatenate 
    df = pd.DataFrame([x], columns=column_names)
   
    l = [data, df]
    data = pd.concat(l)
    #print(data.shape)

x = data.iloc[:,:-1]
Y = data.iloc[:, -1]

c_0 = (Y == 0).sum()
c_1 = (Y == 1).sum()

print(c_0, " ", c_1)

shf_data = data.sample()