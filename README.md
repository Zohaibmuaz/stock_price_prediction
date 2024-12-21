# Apple Stock Price Prediction using Linear Regression
## Overview
This project uses a Linear Regression model to predict the closing price of Apple stock based on features such as opening price, high price, low price, and trading volume. The model is trained and evaluated using historical stock data.
## Dataset
The dataset used in this project is apple_stock_data.csv, which contains the following features:

Features: Open, High, Low, Volume

Target: Close (closing price of the stock)
## Workflow
### Data Loading:
Load the dataset using pandas.

Check for missing values and handle them by dropping rows with NaN values.
### Feature Selection:
Select Open, High, Low, and Volume as input features (X).

Use Close as the target variable (y).
## Data Preprocessing:
Normalize the feature values using StandardScaler to improve model performance.

Split the dataset into training and testing sets (90% training, 10% testing).
## Model Training:
Train a Linear Regression model using the training data.
## Model Evaluation:
Predict the closing prices on the test set.

Calculate the Mean Squared Error (MSE) to evaluate the model's performance.

Display the true vs. predicted labels for the test set.

## Requirements
Make sure to have the following libraries installed:

numpy

pandas

scikit-learn

You can install the required libraries using:
bash


pip install numpy pandas scikit-learn
## How to Run
### Prepare the Dataset:
Ensure the apple_stock_data.csv file is in the same directory as the script.

The dataset should include the columns: Open, High, Low, Volume, and Close.

Run the Script:
bash


python stock_price_prediction.py
## Results
The script will output:

The Mean Squared Error (MSE) of the model.

A DataFrame showing the true and predicted closing prices for the test set.

Example Output


mean_squared_error = 15.23

   true labels   predicted labels

0       145.67            146.12

1       150.34            149.89
...
## Dependencies

numpy: For numerical computations.

pandas: For data manipulation and analysis.

scikit-learn: For preprocessing, model training, and evaluation.
