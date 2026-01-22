"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 02 May 2024
Description: Perform unit test for each function of Random Forest classification.

"""
import numpy as np
import pytest
import os
import sys

decision_tree_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Decision_tree_based_models'))
sys.path.insert(0, decision_tree_dir)
from random_forest_classification import bootstrap_samples, random_feature_subpace, random_forest, predict_random_forest

#===================================================================================================

# Test function to check the generation of bootstrap samples.
def test_bootstrap_samples():
    """
    
    # Purpose of test: Test the functionality of the bootstrap_samples function to generate bootstrap samples(sampling with replacement).

    # Input: Numpy arrays of independent and target training samples and number of boot strapping samples.
            x = np.array([[2, 4], [5, 1], [4, 6], [8, 1], [3, 5]])
            y = np.array([[0], [1], [0], [0], [1]])
            # Number of bootstrap samples
            num_samples, seed  = 3, 10
    Command to run file: pytest test_random_forest.py
    
    # Expected output: Bootstrapped samples must be a subset of all features.

    # Obtained output: The bootstrapped samples are subset of the full features list.

    """
    
    # Sample independent and target features
    x = np.array([[2, 4], [5, 1], [4, 6], [8, 1], [3, 5]])
    y = np.array([[0], [1], [0], [0], [1]])
    # Number of bootstrap samples
    num_samples, seed  = 3, 10
    # Obtain bootstrapped samples by calling the function
    x_bs, y_bs = bootstrap_samples(x, y, num_samples, seed )
    # Verify the shape of the bootstrapped samples and check if they are subset of original features dataset.
    assert(x_bs.shape == (num_samples, x.shape[1]))
    assert(y_bs.shape == (num_samples, y.shape[1]))
    assert((np.all([bs_example in x for bs_example in x_bs])) == True)
    assert((np.all([bs_example in x for bs_example in x_bs])) == True)
#===================================================================================================
# Test function to check the random sub-space of features.
def test_random_feature_subspace():
    """
    
    # Purpose of test: Test the functionality of the random_feature_subpace function to create a feature subspace which is subset of original all features.

    # Input: Numpy arrays of independent samples and number of sub space features to be considered.
            x = np.array([[1, 5, 4, 2], [2, 3, 7, 1], [2, 6, 1, 2], [3, 1, 3, 7], [5, 3, 9, 5]])
            # Number of features to be considered in the feature sub space.
            n_sub_features, seed = 2, 10
            
    Command to run file: pytest test_random_forest.py
    
    # Expected output: feature subspace must be subset of existing all features with replacement.

    # Obtained output: The feature subspace obtained by calling the function random_feature_subpace, such that it match with the expected number of features subset.
                        feature subspace is subset of existing all features with replacement.

    """
    # Sample independent feature
    x = np.array([[1, 5, 4, 2], [2, 3, 7, 1], [2, 6, 1, 2], [3, 1, 3, 7], [5, 3, 9, 5]])
    # Number of features to be considered in the feature sub space.
    n_sub_features, seed = 2, 10
    # Obtain the number of features in subspace by calling the function
    x_sub_features = random_feature_subpace(x, n_sub_features, seed)
    
    # Verify it it consists of expected feature subspace and it should be subset of original feature space.
    assert(x_sub_features.shape[1] == n_sub_features)
    assert(np.all([sub_fea in x.T for sub_fea in x_sub_features.T]) == True)
#===================================================================================================
# Test function to check random forest model.
def test_random_forest():
    """
    
    # Purpose: Test the functionality of the random_forest function to construct a random forest model.

    # Input: Numpy arrays of independent and training features, number of bootstrapped samples, number of sub-features, maximum depth of trees, minimum samples for splitting, and the number of trees in the forest.
            x_train = np.array([[4, 1, 6, 3], [1, 7, 2, 8], [3, 5, 4, 1], [2, 4, 6, 1], [5, 1, 4, 7]])
            y_train = np.array([[1], [0], [1], [1], [0]])
            boot_strapped_samples, num_subfeatures, depth_max, samples_min, num_trees = 3, 2, 3, 2, 2
            seed = 12
    
    Command to run file: pytest test_random_forest.py
    
    # Expected Output: A forest comprising of a given number of decision trees.

    # Obtained Output: The forest which is collection of trees obtained by calling the random_forest function match with the expected number of trees.
  
    """
    # Sample independent and target features
    x_train = np.array([[4, 1, 6, 3], [1, 7, 2, 8], [3, 5, 4, 1], [2, 4, 6, 1], [5, 1, 4, 7]])
    y_train = np.array([[1], [0], [1], [1], [0]])
    boot_strapped_samples, num_subfeatures, depth_max, samples_min, num_trees = 3, 2, 3, 2, 2
    seed = 12
    # Obtain the forest by calling the function
    forest = random_forest(x_train, y_train, boot_strapped_samples, num_subfeatures, depth_max, samples_min, num_trees, seed)
    
    # Verify if the number of trees obtained should match the considered number of trees.
    assert(len(forest) == num_trees)
    
#===================================================================================================
# Test function to predict the class using random forest model.
def test_predict_random_forest():
    """
    
    # Purpose: Test the functionality of the predict_random_forest function to predict the class labels using a random forest model.

    # Input: forest(List of trees) and the test set for prediction.
                forest = [1, (0, 4.0, 1, 0), (0, 3.0, 0, 1), (0, 4.5, 1, 0)]
                x_test = np.array([[1, 2, 3, 6], 
                                   [9, 8, 2, 7], 
                                   [5, 4, 7, 2], 
                                   [4.5, 6, 9, 2]])
                
    Command to run file: pytest test_random_forest.py
    
    # Expected Output: Predicted class labels for the given test set.
                expected_prediction = [1, 0, 0, 1]
    # Obtained Output: The predicted class labels obtained by calling the predict_random_forest function should match with the expected ones.
                            [1, 0, 0, 1]
    """
    # Input the sample forest
    forest = [1, (0, 4.0, 1, 0), (0, 3.0, 0, 1), (0, 4.5, 1, 0)]
    # Sample Test dataset
    x_test = np.array([[1, 2, 3, 6], 
                       [9, 8, 2, 7], 
                       [5, 4, 7, 2], 
                       [4.5, 6, 9, 2]])
    # Obtain the predicted label class by calling the function
    y_pred = predict_random_forest(forest, x_test)
    # Expected class
    expected_prediction = [1, 0, 0, 1]
    
    # Verify whether the obtained class match with the expected class
    assert(expected_prediction == y_pred)
    
#===================================================================================================
    

