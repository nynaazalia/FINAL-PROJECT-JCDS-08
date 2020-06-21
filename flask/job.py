import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import RobustScaler
import joblib


df = pd.read_csv('BankChurn_Clean.csv')

scaler_feature = ['Age','Balance','CreditScore','EstimatedSalary']
scaler = RobustScaler()
df2 = pd.DataFrame(scaler.fit_transform(df[scaler_feature]), columns=scaler_feature)
df2 = pd.concat([df2, df.drop(columns=scaler_feature)], axis=1)

X = df2[['CreditScore', 'Age', 'Balance', 'EstimatedSalary','Tenure', 'NumOfProducts', 'IsActiveMember','France', 'Spain', 'Germany', 'GenderMale']]
# print(X)
Y = df2['Exited']
# print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
# print(X_train)

sm = SMOTE(random_state = 2) 
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train.ravel())


model= KNeighborsClassifier()
model.fit(X_train_smote, y_train_smote.ravel())

joblib.dump(model, 'modelJoblib')
joblib.dump(scaler, 'scaleJoblib')

