# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 00:48:35 2020

@author: Vishal
"""

# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import linear_model


dataset = pd.read_csv('GeodeFinal.csv')
dataset = dataset.fillna(0)
column_names = dataset.columns[4:]

np_arr = dataset.iloc[:, 4:]
#print(type(np_arr))
data = pd.DataFrame(np_arr, columns = column_names)
X = dataset.iloc[:, 4:14].values
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

#################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = 0.4, random_state = 0)

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




#################################################################################




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = 0.3, random_state = 0)
   
from keras.models import Sequential
from keras.layers import Dense
   
#define the model\n",
# nf = no. of features or input variable \n",
# unit1 = no. of units in the hidden layer 1\n",
# unit2 = no. of units in the hidden layer 2 \n",
    
nf = 10 
unit1 = 36
#unit2 = 72
model = Sequential()
model.add(Dense(unit1, input_dim = nf, activation='relu')) 
#model.add(Dense(unit2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
  
# compile the keras model\n",
    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#convert dataframe to numpy array for keras
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# fit the keras model on the dataset\n",
model.fit(X_train, y_train, epochs = 800, batch_size = 30)
   
accuracy = model.evaluate(X_test, y_test)
print('Accuracy: ' , (accuracy*100))
  
y_pred = model.predict_classes(X_test)
   
# draw roc curve as done in other algorithms "
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




#calculate recall maybe and other stuff
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

