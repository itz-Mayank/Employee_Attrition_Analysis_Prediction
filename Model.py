# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE


# Prepare Data
path = r'C:\Users\Mayank Meghwal\Desktop\DS GUVI\Projects\Employee\Preprocessed.csv'
df = pd.read_csv(path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
# df.drop(['StandardHours','EmployeeCount','EmployeeNumber'],axis = 1,inplace=True)

# Distributing Data
X = df.drop(['Attrition_Yes'],axis=1)
Y = df['Attrition_Yes']

# Splitting the data for Train and Test
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
models = [
          DecisionTreeClassifier(),
          RandomForestClassifier()
        ] 

# Standardization to data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Apply SMOTE on scaled data
# sm = SMOTE()
# X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# print("Original y_train distribution:\n", y_train.value_counts())
# print("Resampled y_train distribution:\n", pd.Series(y_train_resampled).value_counts())


# Training the model
for model in models:
  # model.fit(X_train_resampled, y_train_resampled)
  model.fit(X_train, y_train)

  # Predictions
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  # Performance Matrix
  Report = classification_report(y_test,y_pred_test)
  print(f'REPORT OF THE {model} Model : \n',Report)

  score = model.score(X_test,y_test)
  print(f'Score = {score}')

  # Visualization
  residuals = y_test,y_pred_test

  sns.histplot(residuals,kde=True)
  plt.show()
    

input_data = np.array([[0,41,1,1102,0,1,2,2,0,94,3,2,4,0,5993,19479,8,1,11,3,1,0,8,0,1,6,4,0,5,1,0,0,0,0,0,0,0,0,0,0,1,0]])
sample_input_df = pd.DataFrame(input_data, columns=X.columns)
sample_scaled = scaler.transform(sample_input_df)


print(models[0].predict(sample_scaled))