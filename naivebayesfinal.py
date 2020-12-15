# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:21:32 2020

@author: Vishal
"""
##########inporting and dividing####################

# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import shuffle

# Import PySwarms
import pyswarms as ps

dataset = pd.read_csv('GeodFeinal.csv')
dataset = dataset.fillna(0)
column_names = dataset.columns[4:]

np_arr = dataset.iloc[:, 4:]
#print(type(np_arr))
data = pd.DataFrame(np_arr, columns = column_names)
X = dataset.iloc[:, 4:23].values
y = dataset.iloc[:, -1].values


Y = dataset.iloc[:, -1].values
unique, counts = np.unique(Y, return_counts=True)
dict(zip(unique, counts))

X, y = shuffle(X, y)

####################oversampling######################
"""
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

"""


####################### ADASYN ######################



from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = ADASYN().fit_resample(X, y)

clf_smote = LinearSVC().fit(X_resampled, y_resampled)

#dataset division
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 0)
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled, test_size=0.2, random_state = 0)


############### feature scaling
from sklearn.preprocessing import StandardScaler
X_resampled= StandardScaler().fit_transform(X_resampled)
X= StandardScaler().fit_transform(X)

###################### Splitting dataset (after oversampling (redundant now))########################3
# x = oversampled(X)
# Y = oversampled(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = 0.3, random_state = 0)
   


################ keras ###############

from keras.models import Sequential
from keras.layers import Dense
   
#define the model\n",
# nf = no. of features or input variable \n",
# unit1 = no. of units in the hidden layer 1\n",
# unit2 = no. of units in the hidden layer 2 \n",
    
nf = 19
unit1 = 36
#unit2 = 72
model = Sequential()
model.add(Dense(unit1, input_dim = nf, activation='relu')) 
#model.add(Dense(unit2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
  
# compile the keras model\n",
    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   

# fit the keras model on the dataset\n",
model.fit(X_train, y_train, epochs = 100, batch_size = 30)

y_pred = model.predict_classes(X_test)






######################using the actual classifier########################3
#shuffle



# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


#y_pred = "insert something here"


######################## measuring the stats (RESULTS)   (before PSO) ######################
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

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % accuracy)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.3f' % precision)

from sklearn.metrics import f1_score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)


"""
#some more? (not needed)
from sklearn import metrics
print("Accuracy: {}%".format(round(metrics.accuracy_score(y_test, y_pred)*100)))
print("Recall: {}%".format(round(metrics.recall_score(y_test, y_pred)*100)))
print("Precision: {}%".format(round(metrics.precision_score(y_test, y_pred)*100)))
print("F1: {}%".format(round(metrics.f1_score(y_test, y_pred)*100)
"""

##################################### PSO Feature Selection################################

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)

####a neccessary step after oversampling(not needed)
x=x.values
Y=Y.values
Y=Y.astype(np.int64)

####test this (yep this works keep this)(not needed anymore)
X=X_resampled
y=y_resampled


# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 19
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

options = {'c1': 1.72, 'c2': 1.72, 'w':0.9, 'k': 28, 'p':2}
dimensions = 19
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30 , dimensions=dimensions, options=options)
#cost, pos = optimizer.optimize(f, print_step=100, iters=1000, verbose=2)
cost, pos = optimizer.optimize(f, iters=100)



# Create two instances of LogisticRegression
#classifier2 = RandomForestClassifier(max_depth=2, random_state=0)

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

"""
# Perform classification and store performance in P
classifier.fit(X_selected_features, y)

classifier = RandomForestClassifier(max_depth=2, random_state=0)
"""


###################### New splitting of dataset after PSO########################
X_selected_features, y_resampled =  shuffle(X_selected_features, y_resampled)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.2, random_state = 0)
   

######################using the actual classifier (after PSO) ########################3


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


#y_pred = "insert something here"


######################## measuring the stats (RESULTS)   (after PSO) ######################
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


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % accuracy)


from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.3f' % precision)


from sklearn.metrics import f1_score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)

