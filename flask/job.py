import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import RobustScaler
import joblib


data = pd.read_csv('BankChurn_Clean_Scale.csv')
# print(data.head())

X= data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary','France', 'Spain', 'Germany', 'GenderMale']]
# print(X)
Y= data['Exited']
# print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
# print(X_train)

sm = SMOTE(random_state = 2) 
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train.ravel())


model= KNeighborsClassifier()
model.fit(X_train_smote, y_train_smote.ravel())

scaler = RobustScaler()
scaler.fit_transform(data[['CreditScore']])
scaler.fit_transform(data[['Age']])
scaler.fit_transform(data[['Balance']])
scaler.fit_transform(data[['EstimatedSalary']])
scaler.fit_transform(data[['Tenure']])

joblib.dump(model, 'modelJoblib')
joblib.dump(scaler, 'scaleJoblib')

