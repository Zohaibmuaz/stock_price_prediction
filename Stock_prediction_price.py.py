import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


data = pd.read_csv("apple_stock_data.csv")
print(data.head())
feature = ["Open","High","Low","Volume"]
target = "Close"
data = data.dropna()

x = data[feature]
y = data[target]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)
scaler = StandardScaler()
x_train_norm =scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train_norm,y_train)
y_pred= model.predict(x_test_norm)
mse = mean_squared_error(y_test,y_pred)
print(f"mse = {mse}")
# Create a DataFrame to display predictions with corresponding feature values
results = pd.DataFrame(x_test, columns=feature)
results['Actual Close'] = y_test.values
results['Predicted Close'] = y_pred

# Inverse transform the features to their original scale for better understanding
results[feature] = scaler.inverse_transform(results[feature])

print(results)