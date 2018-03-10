# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:49:40 2018

@author: sarath
"""



import pandas as pd
#from sklearn import linear_model

#from sklearn import metrics
from sklearn.cross_validation import train_test_split as splt
from sklearn.metrics import classification_report,confusion_matrix
#from sklearn.ensemble import RandomForestClassifier as rf
import time

#from sklearn import tree
from sklearn import svm




import os
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error   
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

df = pd.read_csv('data.txt', header= None,sep='   ')
df.shape
df.head(5)


features = list(df)
features.remove(17)
label = 17
#==============================================================================
# features= df.copy()
# label=df.copy()
# 
# 
# 
# features = df.iloc[:,:16]
# label = df.iloc[:,16:18]
#==============================================================================

start_time = time.time()

x_train, x_test, y_train, y_test = splt(df[features],df[label],test_size = 0.20,train_size =0.80)


dt = SVR(kernel='linear', C=2, gamma=0.1)
dt.fit(x_train,y_train)

test_data = pd.concat([x_test, y_test], axis=1)

test_data["prediction"] = dt.predict(x_test)

#==============================================================================
# target_names = ['pred 0', 'pred 1']
# print(classification_report(y_test,test_data['prediction'], target_names=target_names))
# 
# 
# print(confusion_matrix(y_test, test_data['prediction']))
# 
#==============================================================================
accuracy=dt.score(x_test,y_test)
print ("Accuracy:", accuracy)

accuracy_list=[]
accuracy_list=accuracy_list.append(accuracy)

print("--- %s seconds ---" % (time.time() - start_time))