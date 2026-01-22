"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 20 Mar 2024
Description: Unit test for regression evaluation metrics.

"""
import numpy as np
import os
import sys
ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'ANN'))
sys.path.insert(0, ANN_dir)
from regression_evaluation_metrics import root_mean_squared_error, mean_absolute_error, r_squared_error

#===============================================================================================================
# Test function to verify the root mean squared error calculation.
def test_root_mean_squared_error():
    """
    
    # Purpose of test: This test make sure that the function root_mean_squared_error, calculates the root mean squared error (RMSE) betweeen the predicted and true values correctly.

    # Input: Numpy arrays of true and predicted values for a set of datapoints.
        y_true = np.array([[3], [4], [6], [8]])
        y_pred = np.array([[2.8], [4.2], [5.7], [8.3]]) 
        
    Command to run file: pytest test_evaluation_metrics_regression.py
    
    # Expected output: The root mean squared error (RMSE) is calculated based on the predicted and true values.
        expected_rmse = 0.2549
    # Obtained output: The root mean squared error (RMSE) value obtained by calling the function root_mean_squared_error, should match with the expected value within a tolerance of 1e-3.
        obtained_rmse = 0.2549
    """
    # True value and prediction value
    y_true = np.array([[3], [4], [6], [8]])
    y_pred = np.array([[2.8], [4.2], [5.7], [8.3]])           
    # Obtain rmse value by calling the function
    obtained_rmse = root_mean_squared_error(y_true, y_pred)
    # Expected root mean square value
    expected_rmse = 0.2549
    
    # Check if the obtained root mean squared value matches the expected value
    assert(expected_rmse - obtained_rmse < 1e-3)
    
#===============================================================================================================
# Test function to verify the mean absolute error calculation.
def test_mean_absolute_error():
    """
    
    # Purpose of test: This test make sure that the function mean_absolute_error, calculates the mean absolute error (MAE) betweeen the predicted and true values correctly.

    # Input: Numpy arrays of true and predicted values for a set of datapoints.
            y_true = np.array([[3], [4], [6], [8]])
            y_pred = np.array([[2.8], [4.2], [5.7], [8.3]])  
            
    Command to run file: pytest test_evaluation_metrics_regression.py
    
    # Expected output: The mean absolute error (MAE) is calculated based on the predicted and true values.
            expected_mae = 0.25
    # Obtained output: The mean absolute error (MAE) value obtained by calling the function mean_absolute_error, should match with the expected value within a tolerance of 1e-3.
            obtained_mae = 0.25
    """
    # True value and prediction value
    y_true = np.array([[3], [4], [6], [8]])
    y_pred = np.array([[2.8], [4.2], [5.7], [8.3]])   
    # Obtain mae value by calling the function
    obtained_mae = mean_absolute_error(y_true, y_pred)
    # Expected mean absolute error value
    expected_mae = 0.25
    
    # Check if the obtained mean absolute error matches the expected value
    assert(expected_mae - obtained_mae < 1e-3)
    
#===============================================================================================================
# Test function to verify the r squared error calculation.
def test_r_squared_error():
    """
    
    # Purpose of test: This test make sure that the function r_squared_error, calculates the R squared error (RSE) betweeen the predicted and true values correctly.

    # Input: Numpy arrays of true and predicted values for a set of datapoints.
            y_true = np.array([[3], [4], [6], [8]])
            y_pred = np.array([[2.8], [4.2], [5.7], [8.3]])
            
    Command to run file: pytest test_evaluation_metrics_regression.py
    
    # Expected output: The R squared error (RSE) is calculated based on the predicted and true values.
            expected_rse = 0.982
    # Obtained output: The R squared error (RSE) value obtained by calling the function r_squared_error, should match with the expected value within a tolerance of 1e-3.
            obtained_rse = 0.982
    """
    # True value and prediction value
    y_true = np.array([[3], [4], [6], [8]])
    y_pred = np.array([[2.8], [4.2], [5.7], [8.3]])   
    # Obtain rse value by calling the function
    obtained_rse = r_squared_error(y_true, y_pred)
    # Expected r squated error value
    expected_rse = 0.982
    
    # Check if the obtained r squared error matches the expected value
    assert(expected_rse - obtained_rse < 1e-3)
#===============================================================================================================
