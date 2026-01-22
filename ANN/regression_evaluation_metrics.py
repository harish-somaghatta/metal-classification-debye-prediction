"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 20 Mar 2024
Description: Evaluation metrics for regression.

"""

import numpy as np

def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the root mean squared error of the model.

    Args:
    y_true (numpy array): True labels.
    y_pred (numpy array): Predicted labels.

    Returns:
    accuracy (float): root mean squared error of the model.
    """
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    return rmse

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error(MAE) between true values and predicted values.

    Args:
    y_true (numpy array): True values.
    y_pred (numpy array): Predicted values.

    Returns:
    accuracy (float): Mean Absolute Error(MAE) of the model.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    
    return mae

def r_squared_error(y_true, y_pred):
    """
    Calculates the R squared error between true values and predicted values.

    Args:
    y_true (numpy array): True values.
    y_pred (numpy array): Predicted values.

    Returns:
    accuracy (float): R squared error of the model.
    """
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - np.mean(y_true))**2)
    
    r_squared = 1 - (num/den)
    
    return r_squared
