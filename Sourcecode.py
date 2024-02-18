# import files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
-----------------------------------------------------------------------------------------------

# Data Collection and Understanding

insurance_data = pd.read_csv("insurance.csv")
----------------------------------------------------
print(insurance_data.isnull().sum())

-----------------------------------------------------------------------------------------------


#Exploratory Data Analysis (EDA) & Univariate analysis

plt.figure(figsize=(10, 6))
sns.histplot(insurance_data['age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
-----------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='charges', data=insurance_data)
plt.title('Insurance Charges by Smoking Status')
plt.xlabel('Smoker')
plt.ylabel('Insurance Charges')
plt.show()

-----------------------------------------------------------------------------------------------

#Bivariate analysis

plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=insurance_data)
plt.title('Insurance Charges by BMI')
plt.xlabel('BMI')
plt.ylabel('Insurance Charges')
plt.show()

-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------

# Model Building

# Convert categorical variables into numerical representations

insurance_data = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'], drop_first=True)

-----------------------------------------------------------------------------------------------

# Split data into features and target variable

X = insurance_data.drop('charges', axis=1)
y = insurance_data['charges']

-----------------------------------------------------------------------------------------------

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

-----------------------------------------------------------------------------------------------

# Train the Linear Regression model

model = LinearRegression()
model.fit(X_train, y_train)

-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------

# Model Evaluation and Interpretation

# Make predictions

y_pred = model.predict(X_test)

-----------------------------------------------------------------------------------------------

# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
---------------------------------------------------
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

-----------------------------------------------------------------------------------------------













