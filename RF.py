import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import joblib
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier


def Train_fold(i,X_train,y_train,X_val,y_val):
    RF = RandomForestClassifier(n_estimators=10,max_features="auto",criterion='gini',oob_score=True)
    RF.fit(X_train, y_train)
    y_pred = RF.predict_proba(X_val)
    AUC = roc_auc_score(y_val[:,0], y_pred[:,0])
    print(AUC)
    joblib.dump(RF, filename='Cancer'+str(i)+'_RF.model')


df=pd.read_table(r'./C-H.txt',sep='\t')
X_list=df.iloc[:,1:-1]
X=np.array(X_list)
y_list=df.iloc[:,-1]
y=np.array(y_list)
Skf=StratifiedKFold(n_splits=10,shuffle=True)
vals=np.array([])
y_vals=np.array([]).reshape(0,3)
y_val_scores=np.array([]).reshape(0,3)
skf1=Skf.split(X,y)
i=0
for train_index, test_index in skf1:
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     y_train = to_categorical(y_train)
     y_test = to_categorical(y_test)
     i=i+1
     Train_fold(i, X_train, y_train, X_test, y_test)






















