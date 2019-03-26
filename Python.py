# Credit Card Fraud 

# Importing Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea


dataset = pd.read_csv("creditcard.csv")

x = dataset.iloc[:,1:30]
y = dataset.iloc[:,30].values
x1 = pd.DataFrame()
x1["Amount"] = dataset["Amount"]

 


from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
x1 =  Scaler.fit_transform(x1)


x = x.drop(["Amount"],axis = 1)
xx = pd.DataFrame(np.c_[x,x1])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(xx,y,train_size = 0.8,random_state = 0)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
rfc.score(x_test,y_test)

# Score is 0.99


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from sklearn.model_selection import GridSearchCV as cv
para = [{"n_estimators" : [30,40,50],"criterion" :["gini","entropy"]},
         {"n_estimators" : [60,70,80,90],"criterion" :["gini","entropy"]}]
cv  = cv(estimator =rfc ,param_grid =para ,cv = 15 )
c= cv.fit(x_train,y_train)
cv.best_params_
cv.best_score_
y_cv = cv.predit(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_cv)