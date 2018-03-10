import os
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error   
from sklearn.grid_search import GridSearchCV

df = pd.read_csv('data.txt', header= None,sep='   ')
df.shape
df.head(5)
start_time = time.time()

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:16],df.iloc[:,16:18],test_size=0.33, random_state=42)

rgr1 = RandomForestRegressor()
rgr1.fit(X_train.iloc[:,:16],y_train.iloc[:,0])
rgr1_pre = rgr1.predict(X_test.iloc[:,:16])
mean_squared_error(y_test.iloc[:,1],rgr1_pre)


print("--- %s seconds ---" % (time.time() - start_time))