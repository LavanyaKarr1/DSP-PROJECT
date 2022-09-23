# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:55:52 2022

@author: Lavanya
"""


import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
stress=pd.read_csv("Stress-Lysis1.csv")
df=pd.DataFrame(stress)
print(df)
print(df.head())
print(df.columns)
print(df.shape)
print(df.corr())
print(df.duplicated().sum())
print(df.describe().T)
print(df.isnull().sum())
print("minimum humidity:",df["Humidity"].min())
print("maximum humidity:",df["Humidity"].max())
print("minimum Temperature:",df["Temperature"].min())
print("maximum Temperature:",df["Temperature"].max())
x=df[["Humidity","Temperature"]]
y=df["Stress Level"]
print(x)
print(y)
plt.plot(df["Humidity"],df["Temperature"])
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print("training data with inputs:","\n",x_train)
print("testing data with inputs:","\n",x_test)
print("training data with outputs:","\n",y_train)
from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
df=pd.DataFrame({"actual":y_test,"predicted":y_pred})
print(df)
from sklearn.metrics import accuracy_score
print("acccuracy:",(accuracy_score(y_test,y_pred)))
data=pd.DataFrame({'original':y_test,'predict':y_pred})
print(data[data["original"]!= data["predict"]])
print("prediction:[Humidity,Temperature]")
s_l=np.array([[20,92]]) 
if(s_l[0,0]<=0 or s_l[0,1]<=0):
    print("not a person")
elif(s_l[0,0]<10):
    print("not  a person")
else:
    predictions=reg.predict(s_l)
    print("stress of the person is: {}".format(predictions))



