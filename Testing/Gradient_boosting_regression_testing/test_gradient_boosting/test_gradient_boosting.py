"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 08 May 2024
Description: Perform unit test   function of Gradient boosting algorithm 

"""
import numpy as np
import os
import sys

decision_tree_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'Decision_tree_based_models'))
sys.path.insert(0, decision_tree_dir)
from gradient_boosting_regression import gradient_boosting_trees, predict_gb

#================================================================================================================================
# Test function to verify the functionality of the gradient_boosting function.
def test_gradient_boosting_trees():
    """
    # Purpose: This test make sure that the function gradient_boosting_trees generates the correct number of decision trees and that each tree is in the expected format.
        
    # Input: Numpy arrays of sample independent and target training dataset, sample number of trees, learning rate, initial prediction, maximum depth of trees and minimum number of samples to split.
            # Sample independent and target training dataset
            x_train = np.array([[4, 2, 5], [11, 6, 2], [10, 4, 8], [5, 7, 22]])
            y_train = np.array([[2.5], [1], [0.75], [3.1]])
            # Sample data
            num_trees, lr, max_depth, min_samples = 5, 0.1, 4, 2
            # Initial prediction based on mean of target values
            f0 = np.mean(y_train)
    
    Command to run file: pytest test_gradient_boosting.py
    
    # Expected output: Number of decision trees generated should be equal to the given number of trees and each tree should be a tuple.

    # Obtained output: The obtained trees from gradient_boosting_trees function matched with the expected ones.
    
    
    """
    # Sample independent and target training dataset
    x_train = np.array([[4, 2, 5], [11, 6, 2], [10, 4, 8], [5, 7, 22]])
    y_train = np.array([[2.5], [1], [0.75], [3.1]])
    # Sample data
    num_trees, lr, max_depth, min_samples = 5, 0.1, 4, 2
    # Initial prediction based on mean of target values
    f0 = np.mean(y_train)
    # Generate gradient boosting trees
    trees = gradient_boosting_trees(x_train, y_train, num_trees, lr, f0, max_depth, min_samples)
    
    # # Verify if the correct number of trees is created
    assert(len(trees) == num_trees)
    # Verify the format of each tree in the list as expected
    for tree in trees:      
        assert(isinstance(tree, tuple))
    
        
#================================================================================================================================

# Test function to verify the functionality of the predict_gb function.
def test_predict_gb():
    """
    
    # Purpose: This test to make that the predict_gb function accurately predicts target values using the gradient boosting regression model.

    # Input: Sample trees, independent test dataset, mean of true values and learning rate.
            trees = [(1, 6.5, (1, 3.0, 0.662, (0, 4.5, -0.837, -1.0875)), 1.2622), (1, 6.5, (1, 3.0, 0.596, (0, 4.5, -0.753, -0.978)), 1.136)]
            # Sample test dataset
            x_test = np.array([[1.1, -3.2, 4], [1.8, 2.6, 2], [4.1, 5, 8], [5, 7, 22]])
            f0, lr = 1.5, 0.1
    
    Command to run file: pytest test_gradient_boosting.py
    
    # Expected Output: The prediction value based on the given sample tree.
            expected_y_hat = [[1.625], [1.625], [1.341], [1.73982]]
    # Obtained Output: The obtained predicted value by calling the function predict_gb, matched with the expected predicted values.

    """
    # Sample trees
    trees = [(1, 6.5, (1, 3.0, 0.662, (0, 4.5, -0.837, -1.0875)), 1.2622), (1, 6.5, (1, 3.0, 0.596, (0, 4.5, -0.753, -0.978)), 1.136)]
    # Sample test dataset
    x_test = np.array([[1.1, -3.2, 4], [1.8, 2.6, 2], [4.1, 5, 8], [5, 7, 22]])
    f0, lr = 1.5, 0.1
    # Obtain the predicted values by calling the function
    obtained_y_hat = predict_gb(x_test, lr, trees, f0)
    # Expected predicted values
    expected_y_hat = [[1.625], [1.625], [1.341], [1.73982]]
    # Verify if th expected prediction values matches obtained prediction values
    for exp_y_hat, obt_y_hat in zip(expected_y_hat, obtained_y_hat):
        assert(exp_y_hat - obt_y_hat < 1e-3)
        
#================================================================================================================================

    