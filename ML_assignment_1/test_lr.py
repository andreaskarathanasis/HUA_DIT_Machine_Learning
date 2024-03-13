# Import necessary modules
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import linear_regression
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Get and downlaod california housing dataset
california_housing_dataset = fetch_california_housing(data_home=None,
                                                      download_if_missing=True,
                                                      return_X_y=False,
                                                      as_frame=False)

# Split predictors and labels into training and testing sub-datasets with seed=42 for reproducability
X_train, X_test, y_train, y_test = train_test_split(california_housing_dataset.data,
                                                    california_housing_dataset.target,
                                                    test_size=0.3,
                                                    random_state=42)


# Create instance of LinearRegression class implemented in linear_regression.py file
custom_model = linear_regression.LinearRegression()

# Call fit function of the model with the training part of our data
custom_model.fit(X_train, y_train)

# Call evaluate function with the testing part our the data
eval, error = custom_model.evaluate(X_test, y_test)

# Calculate and print Root Mean Square Error
RMSE_custom_model = math.sqrt(error)
print(f"RMSE for custom model is: {RMSE_custom_model}")

# Initialize linear regression model from sklearn
sklearn_model = LinearRegression()

# Fit sklearn model on the data 
sklearn_model.fit(X_train, y_train)

# Make predictions
sklearn_preds = sklearn_model.predict(X_test)

# Print RMSE
RMSE_sklearn_model = mean_squared_error(y_true=y_test,y_pred=sklearn_preds,squared=False)
print(f"RMSE for sklearn model is: {RMSE_sklearn_model}")


# Initialize empty array for RMSE values of custom model
RMSE_array_custom_model = []

# Initialize empty array for RMSE values of sklearn model
RMSE_array_sklearn_model = []

# Create loop to run training and testing experiment 20 times
for i in range(20):
  # Create new traing and testing splits every loop
  X_train, X_test, y_train, y_test = train_test_split(california_housing_dataset.data,
                                                      california_housing_dataset.target,
                                                      test_size=0.3,
                                                      shuffle=True)
  # Call custom model's fit function
  custom_model.fit(X_train, y_train)
  # Call custom model's evaluate function 
  eval, error = custom_model.evaluate(X_test, y_test)
  # Calculate RMSE for custom model and save value in appropriate array
  RMSE_array_custom_model.append(math.sqrt(error))


  # Call sklearn model's fit function
  sklearn_model.fit(X_train, y_train)
  # Get predictions of sklearn model
  sklearn_preds = sklearn_model.predict(X_test)
  # Get MSE for sklearn model
  MSE_sklearn_model = mean_squared_error(y_true=y_test,y_pred=sklearn_preds)
  # Calculate RMSE for sklearn model and save value in appropriate array
  RMSE_array_sklearn_model.append(math.sqrt(MSE_sklearn_model))


# Calculate and print mean value of RMSE for custom model
mean_custom_model = np.sum(RMSE_array_custom_model) / len(RMSE_array_custom_model)
print(f"Mean RMSE for custom model is: {mean_custom_model}")

# Calculate and print standard deviation of RMSE values for custom model
SUM= 0
for item in RMSE_array_custom_model :
  SUM += np.square((item - mean_custom_model))

std_custom_model = math.sqrt(SUM/(len(RMSE_array_custom_model)-1))
print(f"Standard deviation for custom model is: {std_custom_model}")


# Calculate and print mean value of RMSE for sklearn model
mean_sklearn_model = np.sum(RMSE_array_sklearn_model) / len(RMSE_array_sklearn_model)
print(f"Mean RMSE for sklearn model is: {mean_sklearn_model}")

# Calculate and print standard deviation of RMSE values for sklearn model.
# In this case the implementation of std from numpy is used,
# since it's already demonstrated previously how it can be calculated manualy
std_sklearn_model = np.std(RMSE_array_sklearn_model)
print(f"Standard deviation for sklearn model is: {std_sklearn_model}")