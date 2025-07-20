# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Prepare Data
path = r'C:\Users\Mayank Meghwal\Desktop\DS GUVI\Projects\Employee\Preprocessed.csv'
df = pd.read_csv(path)
df.dropna(inplace=True)
# df.drop(['StandardHours','EmployeeCount','EmployeeNumber'],axis = 1,inplace=True)

# Distributing Data
X = df.drop(['Attrition_Yes'],axis=1)
y = df['Attrition_Yes']

# Splitting the data for Train and Test
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# Standardization to data
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# Training the model
model = LogisticRegression()
model.fit(x_train,y_train)

# Prediction on train and test data
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Report of the model
Report = classification_report(y_test,y_test_pred)
print(f'REPORT OF THE {model} Model : \n',Report)

# Extracting the model and scalar in pkl format
joblib.dump(model,'C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/Employee.pkl')
print('Model saved successfully...')

joblib.dump(scale, 'C:/Users/Mayank Meghwal/Desktop/DS GUVI/Projects/Employee/scaler.pkl')
print('Scaler saved successfully...')