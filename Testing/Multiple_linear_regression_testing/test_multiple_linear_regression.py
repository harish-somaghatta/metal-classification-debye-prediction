"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 04 May 2024
Description: Perform unit test for each function of Random Forest classification.

"""
import numpy as np
import os
import sys

mlr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Multiple_linear_regression'))
sys.path.insert(0, mlr_dir)
from multiple_linear_regression import parameters_initialization, calculate_gradient, gradient_descent

#=============================================================================================================
#  Test function to verify the correctness of the parameters_initialization function based on the number of features of a training dataset.

def test_parameters_initialization():
    """
    
    # Purpose of test: Test the functionality of test_parameters_initialization function, whether it initializes the parameters correctly.

    # Input: Number of features of a training dataset.
        num_features = 4
    
    #  Command to run file: pytest test_multiple_linear_regression.py

    # Expected output: Shape of weight matrix should be (num_features, 1) and bias should have zero value.

    # Obtained output: The obtained weight matrix shape and bias value matched with the expected ones.
 
    """
    # Number of features of training dataset
    num_features = 4
    
    # Obtain the initialized parameters by calling the function
    obtained_w, obtained_b = parameters_initialization(num_features)
    
    # Expected shape of the weight matrix
    expected_w_shape = (num_features, 1)
    expected_b = 0   # Expected bias value
    
    # Verify whether shape of weight matrix and bias matches expected ones.
    assert(expected_w_shape == obtained_w.shape)
    assert(expected_b == obtained_b)
    
#=============================================================================================================

# Test function to verify the computation of the calculate_gradient function.
def test_calculate_gradient():
    """
    
    # Purpose of test: Test the functionality of calculate_gradient function, whether it computes gradients of error w.r.t parameters correctly.

    # Input: Sample numpy arrays of independent and target features, weights and bias value.
            # Sample independent and dependent features
            x_train = np.array([[1, 2], [4, 3]])
            y_train = np.array([[3], [5]]) 
            # Parameters
            weights = np.ones((x_train.shape[1], 1))
            bias = 1
     Command to run file: pytest test_multiple_linear_regression.py

    # Expected output: The gradients of error w.r.t parameters are to be computed.
            expected_dw = np.array([[6.5], [5.5]])
            expected_db = 2
    # Obtained output: The obtained gradients from the function calculate_gradient should match with the expected ones.
            obtained_dw = np.array([[6.5], [5.5]])
            obtained_db = 2
    """
    # Sample independent and dependent features
    x_train = np.array([[1, 2], [4, 3]])
    y_train = np.array([[3], [5]]) 
    # Parameters
    weights = np.ones((x_train.shape[1], 1))
    bias = 1
    # Obtain the gradients by calling the function
    obtained_dw, obtained_db = calculate_gradient(x_train, y_train, weights, bias)
    # Expected values
    expected_dw = np.array([[6.5], [5.5]])
    expected_db = 2
    # Verify whether the obtained gradients match with the expected gradient values.
    assert(expected_dw - obtained_dw < 1e-3).all()
    assert(expected_db - obtained_db < 1e-3)

#=============================================================================================================
#  Test function to verify that the parameters are updated correctly based on the calculated gradients and learning rate.

def test_gradient_descent():
    """
    
    # Purpose of test: Test the functionality of gradient_descent function, whether it updates the parameters correctly.

    # Input: Sample numpy arrays of weights, gradients of weight and bias value and bias gradient.
            # Initial weights and bias
            w  = np.ones((2, 1))
            b = 1
            # Gradients considered
            dW = np.array([[0.5], [1.3]])
            dB = 1.7
            learning_rate = 0.3 # Learning rate
    
    Command to run file: pytest test_multiple_linear_regression.py

    # Expected output: The parameters are to be updated based on the gradient descent updation.
            expected_w = np.array([[0.85], [0.61]])
            expected_b = 0.49
    # Obtained output: The obtained updated parameters from the function gradient_descent matched with the expected ones.

    """
    # Initial weights and bias
    w  = np.ones((2, 1))
    b = 1
    # Gradients considered
    dW = np.array([[0.5], [1.3]])
    dB = 1.7
    learning_rate = 0.3 # Learning rate
    # Obtained updated parameters by calling the function gradient_descent
    obtained_w, obtained_b = gradient_descent(w, b, dW, dB, learning_rate)
    
    # Expected values
    expected_w = np.array([[0.85], [0.61]])
    expected_b = 0.49
    
    # Verify whether the obtained values and expected values matches.
    assert(expected_w - obtained_w < 1e-3).all()
    assert(expected_b - obtained_b < 1e-3)

#=============================================================================================================    
    