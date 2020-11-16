# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:51:10 2020

@author: masud
"""
#https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids





# Import dataset
df_train=pd.read_csv('creditcard.csv')


### Credit Card Fraud and Non-Fraud ration with graph
target_count = df_train['Class'].value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#target_count.plot(kind='bar', title='Count (target)');

## End of Class ratio

# Here we use Robust Scaler technique for feature scalling
# Scale "Time" and "Amount"



df_train['scaled_amount'] = RobustScaler().fit_transform(df_train['Amount'].values.reshape(-1,1))
df_train['scaled_time'] = RobustScaler().fit_transform(df_train['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df_train.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()

# Define the prep_data function to extrac features 
def prep_data(df):
    X = df.drop(['Class'],axis=1, inplace=False)  
    X = np.array(X).astype(np.float)
    y = df[['Class']]  
    y = np.array(y).astype(np.float)
    return X,y

# Create X and y from the prep_data function 
X, y = prep_data(df_scaled)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print(X_train.shape)

# ****** LogisticRegression Accuration test
##model = LogisticRegression()
##model.fit(X_train, y_train)
##y_pred = model.predict(X_test)

##accuracy = accuracy_score(y_test, y_pred)
##print("Accuracy: %.2f%%" % (accuracy * 100.0))
# End of accuracy test


# Random UnderSampling
undersam = RandomOverSampler()
# resample the training data
X_undersam, y_undersam = undersam.fit_sample(X_train,y_train)

#After resampling again accuracy count
model = RandomForestClassifier()
model.fit(X_undersam, y_undersam)
y_pred_under = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred_under)

print("============ Random Forest Classifier ================%")
print("Accuracy After RandomUnderSampler: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test, y_pred_under)
print("Accuracy After ROC: %.2f%%" % (roc_auc * 100.0))
#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_under)
pre_scor= precision_score(y_test, y_pred_under)
re_scor = recall_score(y_test, y_pred_under)
f1_scor = f1_score(y_test, y_pred_under)
##print("\n ROC AUC Score:  %.2f%%" % (roc_auc * 100.0))
print("Precision Score:  %.2f%%" % (pre_scor * 100.0))
print("Recall Score:  %.2f%%" % (re_scor * 100.0))
print('F1-Measure: %.2f%%' % (f1_scor * 100.0))

############################################
# Class count
# Define the prep_data function to extrac features 
def prep_data1(df):
    X1 = df[:, :-1]
    y1 = df[:, -1] 
    return X1,y1

# Create X and y from the prep_data function 
X, y = prep_data(df_scaled)
y= y.astype(np.int64)
#print(Counter(y))

df_ar_x = pd.DataFrame(X_train)
df_ar_y = pd.DataFrame(y_train)
df_xy=pd.concat([df_ar_x,df_ar_y],axis=1)



#var i=0
i=0
two_split = np.array_split(df_xy, 2)

data1 = np.array(two_split[0]).astype(np.float)
X1 , y1 = prep_data1(data1)
#----------
df1 = pd.DataFrame(y1, columns = ['Class'])
target_count1 = df1['Class'].value_counts()
print('Class-1 0:', target_count1[0])
print('Class-1 1:', target_count1[1])
print('Proportion-1:', round(target_count1[0] / target_count1[1], 0), ': 1')
#-------------
# Random UnderSampling
#RandomUnderSampler
#DecisionTreeClassifier
over1 = RandomOverSampler()
# resample the training data
X_over1, y_over1 = over1.fit_sample(X1,y1)

#----------
df2 = pd.DataFrame(y_over1, columns = ['Class'])
target_count2 = df2['Class'].value_counts()
print('Class-2 0:', target_count2[0])
print('Class-2 1:', target_count2[1])
print('After OverSampling Proportion:', round(target_count2[0] / target_count2[1], 0), ': 1')
#-------------
#After resampling again accuracy count
model = RandomForestClassifier()
model.fit(X_over1, y_over1)
y_pred_over1 = model.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred_over1)
#print("Accuracy After RandomOverSlice-1: %.2f%%" % (accuracy1 * 100.0))

roc_auc1 = roc_auc_score(y_test, y_pred_over1)
#print("Accuracy After ROC-1: %.2f%%" % (roc_auc1 * 100.0))

pre_scor1= precision_score(y_test, y_pred_over1)
re_scor1 = recall_score(y_test, y_pred_over1)
f1_scor1 = f1_score(y_test, y_pred_over1)
##print("\n ROC AUC Score:  %.2f%%" % (roc_auc * 100.0))
#print("\n Precision Score-1:  %.2f%%" % (pre_scor1 * 100.0))
#print("\n Recall Score-1:  %.2f%%" % (re_scor1 * 100.0))
#print('\n F1-Measure-1: %.2f%%' % (f1_scor1 * 100.0))
#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_over1)


data2 = np.array(two_split[1]).astype(np.float)
X2, y2 = prep_data1(data2)

# Random UnderSampling
over2 = RandomUnderSampler()
# resample the training data
X_over2, y_over2 = over2.fit_sample(X2,y2)

#After resampling again accuracy count
model = RandomForestClassifier()
model.fit(X_over2, y_over2)
y_pred_over2 = model.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred_over2)
print("Accuracy After RandomOverSlice-2: %.2f%%" % (((accuracy1+accuracy2)/2) * 100.0))

roc_auc2 = roc_auc_score(y_test, y_pred_over2)
print("Accuracy After ROC-2: %.2f%%" % ((roc_auc1+roc_auc2)/2 * 100.0))

pre_scor2= precision_score(y_test, y_pred_over2)
re_scor2 = recall_score(y_test, y_pred_over2)
f1_scor2 = f1_score(y_test, y_pred_over2)
##print("\n ROC AUC Score:  %.2f%%" % (roc_auc * 100.0))
print("\n Precision Score-2:  %.2f%%" % ((pre_scor1+pre_scor2)/2 * 100.0))
print("\n Recall Score-2:  %.2f%%" % ((re_scor1+re_scor2)/2 * 100.0))
print('\n F1-Measure-2: %.2f%%' % ((f1_scor1+f1_scor2)/2 * 100.0))
#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_over2)
#######################################

#for array in two_split:       
 #    array1 = array
   #  X_sp, y_sp = prep_over(array)
   
   
 
#print('X5',X_test3.shape)

#print('Class 0:', af_over1[0])
#print('Class 1:', af_over1[1])
#print('Proportion:', round(af_over1[0] / af_over1[1], 0), ': 1')
# Obtain precision and recall

#/* 
##precision, recall, thresholds = precision_recall_curve(y_test, y_pred_under)

##print('Classifcation report UnderSampler:\n', classification_report(y_test, y_pred_under))

##print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_under))

# Calculate Area Under the Receiver Operating Characteristic Curve
##roc_auc = roc_auc_score(y_test, y_pred_under)

##pre_scor= precision_score(y_test, y_pred_under)
##re_scor = recall_score(y_test, y_pred_under)
##f1_scor = f1_score(y_test, y_pred_under)
##print("\n ROC AUC Score:  %.2f%%" % (roc_auc * 100.0))
##print("\n Precision Score:  %.2f%%" % (pre_scor * 100.0))
##print("\n Recall Score:  %.2f%%" % (re_scor * 100.0))
##print('\n F1-Measure: %.2f%%' % (f1_scor * 100.0))
#*/
#TN|FP
#FN|TP
#Precision=TP/(TP+FP)
#Recall=TP/(TP+FN)
#F1=2(Precision*Recall)/(precision+Recall)