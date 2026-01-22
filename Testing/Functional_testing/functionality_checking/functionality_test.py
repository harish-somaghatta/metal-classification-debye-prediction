# Functionality test
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 02 May 2024
Description: Functional test for classification and regression models.

"""
import os
import sys
ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..',  'ANN'))
sys.path.insert(0, ANN_dir)
knn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..',  'knn'))
sys.path.insert(0, knn_dir)
dt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..',  'Decision_tree_based_models'))
sys.path.insert(0, dt_dir)
mlr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..',  'Multiple_linear_regression'))
sys.path.insert(0, mlr_dir)

from knn_classification import main_knn
from multiple_linear_regression import mlr_main
from random_forest_classification import main_random_forest
from gradient_boosting_regression import gradient_boosting_main
from ANN_regression import main_reg
from dataset_functional import classification_dataset, regression_dataset
from ANN_classification import ANN_classification_main

import pytest
import numpy as np
np.random.seed(10)

#==================================================================================================================================


# Test function to verify the predicted labels for KNN classification algorithm.

def test_integrated_knn():
    """
    # Purpose of test: To test the functionality of knn main function.

    # Input: A file containing independent and dependent features.
            x_train = np.array([[1, 8], [5, 4], [6, 2], [5, 6], [3, 8], [7, 9], [8, 2], [2, 4]])
            y_train = np.array([[0], [1], [1], [0], [1], [0], [1], [0]])
            x_test = np.array([[9,1], [3, 2]])
            y_test = np.array([1, 1])
            
    Command to run file: pytest functionality_test.py
    
    # Expected outpuut: The model should choose the label based on the euclidean distance between each test datapoint and all train datapoints. ([0.0, 1.0])

    # Obtained output: [1.0, 1.0]
    
    """
    # Get the predicted labels by calling the function
    x_train = np.array([[1, 8], [5, 4], [6, 2], [5, 6], [3, 8], [7, 9], [8, 2], [2, 4]])
    y_train = np.array([[0], [1], [1], [0], [1], [0], [1], [0]])
    x_test = np.array([[9,1], [3, 2]])
    y_test = np.array([1, 1])
    predict, _ = main_knn(x_train, x_test, y_train, y_test, 3)
    # Expected output
    expected_output = y_test
    
    # Verify whether expected and obtained output matches
    assert(predict == expected_output).all()
    
#==================================================================================================================================
#  Test function to verify the predicted values for ANN Regression.

def test_integrated_ANN_regression():
    """
    # Purpose of test: To test the functionality of main function ANN regression(main_reg).

    # Input: Training dataset - Independent features(x_train) with shape (49, 2), each row contains integers from 1 to 49 and 2. 
    #                           Dependent feature(y_train) with shape (49, 1), each row contains integers from 1 to 49 multiplied by 2. 
    #        Test dataset   - Independent features(x_test) An array with shape (4, 2) - [20, 2], [15, 2], [11, 2], [5, 2]
    #        Hyperparameters - number of epoch(10000), Mini-batch size (m_batch) is 2, Learning rate (alpha) is 0.354728214309162, Number of neurons (n_neurons) in hidden layers is [50, 50] 
    
    Command to run file: pytest functionality_test.py
    
    # Expected output: The expected_values are the same as y_train with shape (49, 1), each row contains integers from 1 to 49 multiplied by 2.
    #                   and y_test, which are [[40], [30], [22], [10]].

    # Obtained output: (y_train - expected_train) is within tolerance of  1e-2 and
    #    ([[40], [30], [22], [10]] - predicted value) is within tolerance of 1e-2.
    """
    # Initialize hyperparameters
    epoch, m_batch, alpha = 10000, 2, 0.354728214309162
    n_neurons = [50, 50]
    # Get training and test datasets by calling the function
    x_train, y_train, x_test, y_test = regression_dataset()
    # Expected train and test dataset values
    expected_train_val = np.round(y_train, decimals=2)
    expected_test_val = np.round(y_test, decimals=2)
    # Get the train and test prediction values by calling the function
    predicted_train, predicted_test = main_reg(x_train, x_test, y_train, y_test, epoch, alpha, m_batch, n_neurons)
    
    # Verify that the predicted values are close to the expected values within a tolerance of 1e-2
    assert(expected_train_val - predicted_train < 1e-1).all()
    assert(expected_test_val - predicted_test < 1e-1).all()

#==================================================================================================================================
    


def test_integrated_multiple_linear_regression():
    """
    # Purpose of test: To test the functionality of main function Multiple Linear Regression algorithm(mlr_main).

    # Input: Training dataset - Independent features(x_train) with shape (49, 2), each row contains integers from 1 to 49 and 2. 
    #                           Dependent feature(y_train) with shape (49, 1), each row contains integers from 1 to 49 multiplied by 2. 
    #        Test dataset   - Independent features(x_test) An array with shape (4, 2) - [20, 2], [15, 2], [11, 2], [5, 2]
    #        Hyperparameters - number of iterations(15000)
    
    Command to run file: pytest functionality_test.py

    # Expected output: The expected_values are the same as y_train with shape (49, 1), each row contains integers from 1 to 49 multiplied by 2.
    #                   and y_test, which are [[40], [30], [22], [10]].

    # Obtained output: (y_train - expected_train) is within tolerance of  1e-2 and
    #    ([[40], [30], [22], [10]] - predicted value) is within tolerance of 1e-2.
    """
    # Get training and test datasets by calling the function
    x_train, y_train, x_test, y_test = regression_dataset()
    n_iterations = 15000
    # Expected train and test dataset values
    expected_train_val = np.round(y_train, decimals=2)
    expected_test_val = np.round(y_test, decimals=2)
    # Get the train and test prediction values by calling the function
    predicted_train, predicted_test = mlr_main(x_train, x_test, y_train, y_test, n_iterations)
    
    # Verify that the predicted values are close to the expected values within a tolerance of 1e-2
    assert(expected_train_val - predicted_train < 1e-1).all()
    assert(expected_test_val - predicted_test < 1e-1).all()
    
#==================================================================================================================================

def test_integrated_gradient_boosting():
    """
    Test function to verify the predicted values for Gradient Boosting regression algorithm.
    # Purpose of test: To test the functionality of Gradient Boosting Regression main function(gradient_boosting_main).

    # Input: Training dataset - Independent features(x_train) with shape (49, 2), each row contains integers from 1 to 49 and 2. 
    #                           Dependent feature(y_train) with shape (49, 1), each row contains integers from 1 to 49 multiplied by 2. 
    #        Test dataset   - Independent features(x_test) An array with shape (4, 2) - [20, 2], [16, 2], [11, 2], [5, 2]
    #        Hyperparameters - number of trees(150), The maximum depth of each tree (100), The minimum number of samples required to split an internal node (1).
    
    Command to run file: pytest functionality_test.py


    # Expected output: The expected_values are the same as y_test, which are [[40], [32], [22], [10]].

    # Obtained output: [[40], [30], [22], [10]] - predicted value should be within tolerance of  1e-2.

    
    """
    # Give hyperparameters
    num_trees, max_depth, min_samples = 150, 100, 1
    
    # Get training and test datasets by calling the function
    x_train, y_train, x_test, y_test = regression_dataset()
    # Expected as test datapoints
    expected_values = np.round(y_test, decimals=2)
    
    # Call the gradient boosting function to get the predicted values
    predicted_values, _, _, _ = gradient_boosting_main(x_train, x_test, y_train, y_test, num_trees, max_depth, min_samples)
    
    # Verify that the predicted values are close to the expected values within a tolerance of 1e-2
    assert(predicted_values - expected_values < 1e-2).all()
    
#==================================================================================================================================

    
def test_integrated_ANN_classification():
    
    """
    Purpose of test: Test function to verify the predicted values for ANN Classification.
    # Input: Values are taken from classification_dataset function in dataset_functional file.
        Training dataset - Independent features(x_train) of training dataset.
    #                           Dependent feature(y_train) of training dataset.
    #        Test dataset   - Independent features(x_test) of test dataset
    #        Hyperparameters - number of epoch(15000), Mini-batch size (m_batch) is 2, Learning rate (alpha) is 0.0-8, Number of neurons (n_neurons) in hidden layers is [9] 
    
    Command to run file: pytest functionality_test.py
    
    # Expected output: The expected_values are the same as y_train and y_test.

    # Obtained output: (y_train - expected_train) are equal.
    #    ([[1.], [1.], [0.], [0.], [1.], [1.], [0.], [0.]] - predicted value) are equal
    
    """
    # Considering hyperparameters
    epoch,m_batch, alpha = 15000, 2, 0.08
    
    n_neurons = [9]
    
    # Get training and test datasets by calling the function
    x_train, y_train, x_test, y_test = classification_dataset()
    target_regression = np.zeros(x_test.shape)
    # Expected train and test dataset values
    expected_train_val = y_train
    expected_test_val = y_test
    # Get the train and test prediction values by calling the function
    predicted_train, predicted_test = ANN_classification_main(x_train, x_test, y_train, y_test, epoch, alpha, m_batch, n_neurons, target_regression)
    
    # Verify that the predicted values are same as the expected values 
    assert(expected_train_val == predicted_train ).all()
    assert(expected_test_val == predicted_test).all()

#==================================================================================================================================
    
    