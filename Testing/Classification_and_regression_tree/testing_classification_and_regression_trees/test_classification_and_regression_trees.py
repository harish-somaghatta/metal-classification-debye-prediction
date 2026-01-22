"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 22 Apr 2024
Description: CART(Classification and Regression Trees) for Random Forest classification and Gradient Boosting regression.

"""
import numpy as np
import pytest
import os
import sys

decision_tree_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'Decision_tree_based_models'))
sys.path.insert(0, decision_tree_dir)
from classification_and_regression_tree import tree_leaf, calculate_impurity, best_values, split_dataset_, decision_tree, classify_example, predict_val


# Test the function 'tree_leaf' to check the classification label calculated for leaf node.
def test_leaf_classification():
    """
    Purpose of test: Test the function 'tree_leaf' which classifies the label based on maximum count.
    
    Input: Target feature with class labels (0's and 1's) and the model type as classification.
            target_feature = np.array([1, 1, 0, 1, 1, 0])
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: The function should return a class with maximum count (Here 1).
            expected_label = 1
    Obtained output: By taking target feature as input, should output 1 based on maximum occurrence.
            label = 1
    """
    # Input target values
    target_feature = np.array([1, 1, 0, 1, 1, 0])
    expected_label = 1   # Expected output
    # Obtained output by calling the function
    obtained_label = tree_leaf(target_feature, "classification")
    
    # Verify whether the obtained output is close to the expected output
    assert(expected_label == obtained_label)


# Test the function 'tree_leaf' to check the predicted value calculated for leaf node.
def test_leaf_regression():
    """
    Purpose of test: Test the functionality of tree_leaf function which predicts the value based on maximum count.
    
    Input: Target feature with target values.
            target_feature = np.array([2, 3, 1, 4, 6, 2])
            
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: The function should return the mean value as the predicted value.
            expected_label = 3 
    Obtained output: By taking target feature values as input, tree_leaf should output the mean value as the predicted value.
            label = 3 
    """
    # Input target values
    target_feature = np.array([2, 3, 1, 4, 6, 2])
    expected_label = 3   # Expected output
    # Obtained output by calling the function
    obtained_label = tree_leaf(target_feature, "regression")
    
    # Verify whether the obtained output is close to the expected output
    assert(expected_label == obtained_label)


# 
def test_classification_gini_impurity():
    """
    Purpose of test: Test the functionality of calculate_impurity function which computes the gini impurity for classification model.
    
    Input: Target feature with target values.
            target_label = np.array([1, 1, 0, 1, 1, 0, 0, 0])
            
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: The function should return the gini impurity value.
            expected_output = 0.5 
    Obtained output: By taking target feature values as input, calculate_impurity should output the gini impurity value.
            output = 0.5
    """
    # Input target values
    target_label = np.array([1, 1, 0, 1, 1, 0, 0, 0])
    expected_output = 0.5   # Expected output
    # Obtained output by computing gini impurity calling the function
    obtained_output = calculate_impurity(target_label, "classification")
    
    # Verify whether the obtained output is close to the expected output
    assert(expected_output == obtained_output)


# Test the functionality of calculate_impurity function to verify whether the calculation of Mean Squared Error (MSE) for regression is correct.
def test_regression_mse():
    """
	Purpose of test: Test the functionality of calculate_impurity function to verify whether the calculation of Mean Squared Error (MSE) for regression is correct.
    
    Input: An array of target values.
            target_values = np.array([2.5, 1.5, 2.5, 3, 1.5, 8])
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: The expected Mean Squared Error value.
            expected_output = 4.972
    Obtained output: The calculated MSE value calling the function calculate_impurity.
            output = 4.972
    """
    # Input target values
    target_values = np.array([2.5, 1.5, 2.5, 3, 1.5, 8])
    expected_output = 4.972 # Expected output for MSE
    # Obtained output by computing MSE calling the function
    obtained_output = calculate_impurity(target_values, "regression")
    
    # Verify whether the obtained output is close to the expected output
    assert(expected_output - obtained_output < 1e-3) == True


def test_best_val_classification():
    """
	Purpose of test: Test the best_values function to verify whether it determines the best split, best feature index, and best impurity gain value for classification.
		
    Input: Simple numpy arrays of independent and target training values.
            independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 0.7], [4.5, 0.9]])
            target_feature = np.array([0, 0, 0, 1])
            
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: Best split, best feature index, and best impurity gain value for classification.
            expected_best_split_val = 3.25
            expected_feature_idx = 0
            expected_gain = 0.125
    Obtained output: The obtained values should match the expected values.
            obtained_best_split_val = 3.25
            obtained_feature_idx = 0
            obtained_gain = 0.125
    """
    # Independent features and target feature.
    independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 0.7], [4.5, 0.9]])
    target_feature = np.array([0, 0, 0, 1])
    # Expected best split value, feature index, and impurity gain.
    expected_best_split_val = 3.25
    expected_feature_idx = 0
    expected_gain = 0.125
    # Obtained the best values by calling the function
    obtained_best_split_val, obtained_feature_idx, obtained_gain = best_values(independent_features, target_feature, "classification")
        
    # Verify if the obtained values match the expected values
    assert(expected_best_split_val == obtained_best_split_val)
    assert(expected_feature_idx == obtained_feature_idx)
    assert(expected_gain == obtained_gain)


def test_best_val_regression():
    """
	Purpose of test:Test the best_values function to verify whether it determines the best split, best feature index, and best impurity gain value for regression.
		
    Input: Simple numpy arrays of independent and target training values.
            independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
            target_feature = np.array([5, 1, 1.5, 8])
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: Best split, best feature index, and best impurity gain value for regression.
            expected_best_split_val = 1.45
            expected_feature_idx = 1
            expected_gain = 5.671
    Obtained output: The obtained values should match the expected values.
            obtained_best_split_val = 1.45
            obtained_feature_idx = 1
            obtained_gain = 5.671
    
    """
    # Independent features and target feature.
    independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
    target_feature = np.array([5, 1, 1.5, 8])
    # Expected best split value, feature index, and information gain.
    expected_best_split_val = 1.45
    expected_feature_idx = 1
    expected_gain = 5.671
    # Obtained the best values by calling the function
    obtained_best_split_val, obtained_feature_idx, obtained_gain = best_values(independent_features, target_feature, "regression")
    
    # Verify if the obtained values match the expected values.
    assert(expected_best_split_val - obtained_best_split_val < 1e-3) == True
    assert(expected_feature_idx - obtained_feature_idx < 1e-3) == True
    assert(expected_gain - obtained_gain < 1e-3) == True


def test_split_dataset_classification():
    """
	Purpose of test: Test the split_dataset function to verify whether it splits the dataset correctly based on the best split value and feature index for classification.
    
    Input: Simple numpy arrays of independent and target training values, and best split and feature index values.
        independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
        target_feature = np.array([0, 0, 0, 1])
        best_split_val, best_feature_idx = 3.25, 0
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: Splitted left node and right node.
        expected_left_node_fea = np.array([[2.0, 5.3], [1.3, 3.5]])
        expected_right_node_fea = np.array([[6.0, 1.7], [4.5, 1.2]])
        expected_left_node_tar = [0, 0]
        expected_right_node_tar = [0, 1]
    Obtained output: The obtained splitted node values should match the expected ones.
        obtained_left_node_fea = np.array([[2.0, 5.3], [1.3, 3.5]])
        obtained_right_node_fea = np.array([[6.0, 1.7], [4.5, 1.2]])
        obtained_left_node_tar = [0, 0]
        obtained_right_node_tar = [0, 1]
    """
    # Independent features and target feature.
    independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
    target_feature = np.array([0, 0, 0, 1])
    # Best split and feature index values
    best_split_val, best_feature_idx = 3.25, 0
    
    # Expected node values
    expected_left_node_fea = np.array([[2.0, 5.3], [1.3, 3.5]])
    expected_right_node_fea = np.array([[6.0, 1.7], [4.5, 1.2]])
    expected_left_node_tar = [0, 0]
    expected_right_node_tar = [0, 1]
    # Obtain the splitted node values by calling the function
    obtained_left_node_fea, obtained_right_node_fea, obtained_left_node_tar, obtained_right_node_tar = split_dataset_(independent_features, target_feature, best_split_val, best_feature_idx, "classification")
    
    # Verify the node values with expected ones
    assert(expected_left_node_fea == obtained_left_node_fea).all()
    assert(expected_right_node_fea == obtained_right_node_fea).all()
    assert(expected_left_node_tar == obtained_left_node_tar).all()
    assert(expected_right_node_tar == obtained_right_node_tar).all()


def test_split_dataset_regression():
    """
	Purpose of test: Test the split_dataset function to verify whether it splits the dataset correctly based on the best split value and feature index for regression.
    
    Input: Simple numpy arrays of independent and target training values, and best split and feature index values.
        independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
        target_feature = np.array([5, 1, 1.5, 8])
        best_split_val, best_feature_idx = 1.45, 1
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: Splitted left node and right node.
        expected_left_node_fea = np.array([[4.5, 1.2]])
        expected_right_node_fea = np.array([[2, 5.3], [1.3, 3.5],[6,  1.7]])
        expected_left_node_tar = [8]
        expected_right_node_tar =  [5, 1, 1.5]
    Obtained output: The obtained splitted node values should match the expected ones.
        obtained_left_node_fea = np.array([[4.5, 1.2]])
        obtained_right_node_fea = np.array([[2, 5.3], [1.3, 3.5],[6,  1.7]])
        obtained_left_node_tar = [8]
        obtained_right_node_tar =  [5, 1, 1.5]
    """
    # Independent features and target feature.
    independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
    target_feature = np.array([5, 1, 1.5, 8])
    # Best split and feature index values
    best_split_val, best_feature_idx = 1.45, 1
    # Expected node values
    expected_left_node_fea = np.array([[4.5, 1.2]])
    expected_right_node_fea = np.array([[2, 5.3], [1.3, 3.5],[6,  1.7]])
    expected_left_node_tar = [8]
    expected_right_node_tar =  [5, 1, 1.5]
    # Obtain the splitted node values by calling the function
    obtained_left_node_fea, obtained_right_node_fea, obtained_left_node_tar, obtained_right_node_tar = split_dataset_(independent_features, target_feature, best_split_val, best_feature_idx, "regression")
    
    # Verify the node values with expected ones
    assert(expected_left_node_fea == obtained_left_node_fea).all()
    assert(expected_right_node_fea == obtained_right_node_fea).all()
    assert(expected_left_node_tar == obtained_left_node_tar).all()
    assert(expected_right_node_tar == obtained_right_node_tar).all()


def test_decision_tree_classification():
    """
	Purpose of test: Test the decision_tree function for classification to verify whether the function returns the expected tuple containing the feature index, split value, left subtree, and right subtree correctly.
    
    Input: Simple numpy arrays of independent and target training values with maximum depth of the tree and minimum number of samples required to perform split.
            independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
            target_feature = np.array([0, 0, 0, 1])
            # Maximum depth of tree and minimum number of samples for split
            max_depth, min_samples = 1, 1
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: Tuple containing the feature index, split value, left subtree, and right subtree.
            expected_tuple = (1, 1.45, 1, 0)
    Obtained output: The obtained tuple values should match the expected ones.
            obtained_tuple = (1, 1.45, 1, 0)
    """
    # Independent features and target feature.
    independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
    target_feature = np.array([0, 0, 0, 1])
    # Maximum depth of tree and minimum number of samples for split
    max_depth, min_samples = 1, 1
    # expected_tuple - (feature_idx, split_val , left_tree, right_tree)
    expected_tuple = (1, 1.45, 1, 0)
    obtained_tuple = decision_tree(independent_features, target_feature, max_depth, "classification", min_samples)
    
    # Verify the tuple values with expected ones
    assert(expected_tuple == obtained_tuple)


def test_decision_tree_regression():
    """
	Test the decision_tree function for regression to verify whether the function returns the expected tuple containing the feature index, split value, left subtree, and right subtree correctly.
    
    Input: Simple numpy arrays of independent and target training values with maximum depth of the tree and minimum number of samples required to perform split.
            independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
            target_feature = np.array([5, 1, 1.5, 8])
            # Maximum depth of tree and minimum number of samples for split
            max_depth, min_samples = 1, 1
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: Tuple containing the feature index, split value, left subtree, and right subtree.
            # expected_tuple - (feature_idx, split_val , left_tree, right_tree)
            expected_tuple = (1, 1.45, 8.0, 2.5)
    Obtained output: The obtained tuple values should match the expected ones.
            obtained_tuple = (1, 1.45, 8.0, 2.5)
    """
    # Independent features and target feature.
    independent_features = np.array([[2, 5.3], [1.3, 3.5], [6, 1.7], [4.5, 1.2]])
    target_feature = np.array([5, 1, 1.5, 8])
    # Maximum depth of tree and minimum number of samples for split
    max_depth, min_samples = 1, 1
    # expected_tuple - (feature_idx, split_val , left_tree, right_tree)
    expected_tuple = (1, 1.45, 8.0, 2.5)
    obtained_tuple = decision_tree(independent_features, target_feature, max_depth, "regression", min_samples)
    
    # Verify the tuple values with expected ones
    assert(expected_tuple == obtained_tuple)


def test_classify_example_right_left_node():
    """
	Purpose of test: Test the classify_example function for classification and regression to verify whether the function correctly classifies based on the decision tree.
		
    Input: Simple example and tree to classify.
            example = np.array([2.4, 1.2])
            # Classification tree
            tree_classification = ((1, 1.85, (0, 1.2, 0, 1), (1, 2.125, 1, 1)))
            # Regression tree
            tree_regression = ((1, 1.85, (0, 1.2, 2.5, 1.5), (1, 2.125, 4.5, 6.0)))
    
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: The classification tree to classify the label and regression tree to predict the value.
            expected_class_label = 1
            expected_prediction_val = 1.5
    Obtained output: The obtained label and prediction value should match the expected ones.
            obtained_class_label = 1       
            obtained_prediction_val = 1.5
            
    """
    example = np.array([2.4, 1.2])
    # Classification tree
    tree_classification = ((1, 1.85, (0, 1.2, 0, 1), (1, 2.125, 1, 1)))
    # Expected class label
    expected_class_label = 1
    # Obtained class label by calling the function
    obtained_class_label = classify_example(example, tree_classification)
    
    # Regression tree
    tree_regression = ((1, 1.85, (0, 1.2, 2.5, 1.5), (1, 2.125, 4.5, 6.0)))
    # Expected prediction value
    expected_prediction_val = 1.5
    # Obtained prediction value by calling the function
    obtained_prediction_val = classify_example(example, tree_regression)
    
    # Verify if the obtained class label and prediction value match the expected ones
    assert(expected_class_label == obtained_class_label)
    assert(expected_prediction_val == obtained_prediction_val)


def test_predict_val_leaf_node():
    """
	Purpose of test: Test the functionality of the predict_val function to predict class labels for classification trees and values for regression trees.
    
    Input: An array of example and, classification and regression tree.
            x = np.array([[8, 2.3], [1.3, 1.7], [6, 1.9], [0.5, 1.2]])
            # Classification tree
            classification_tree = (1, 1.85, 0, 1)
           # Regression tree
           regression_tree = ((1, 1.85, (0, 1.2, 2.5, 1.5), (1, 2.125, 4.5, 6.0)))
           
    Command to run file: pytest test_classification_and_regression_trees.py
    
    Expected output: For classification, the expected class labels. For regression, the expected prediction values.
            # Expected output for classification
            expected_output_classification = [1, 0, 1, 0]
            # Expected output for regression
            expected_output_regression = [6.0, 1.5, 4.5, 2.5]
    Obtained output: The obtained class labels or prediction values by calling the predict_val function should match the expected ones.
            obtained_output_classification = [1, 0, 1, 0]
            obtained_output_regression = [6.0, 1.5, 4.5, 2.5]
    """
    # Example
    x = np.array([[8, 2.3], [1.3, 1.7], [6, 1.9], [0.5, 1.2]])
    # Classification tree
    classification_tree = (1, 1.85, 0, 1)
    # Expected output for classification
    expected_output_classification = [1, 0, 1, 0]
    # Obtained output for classification by calling the function
    obtained_output_classification = predict_val(x, classification_tree)
    
    # Regression tree
    regression_tree = ((1, 1.85, (0, 1.2, 2.5, 1.5), (1, 2.125, 4.5, 6.0)))
    # Expected output for regression
    expected_output_regression = [6.0, 1.5, 4.5, 2.5]
    # Obtained output for regression by calling the function
    obtained_output_regression = predict_val(x, regression_tree)
    
    # Verify if the obtained class label and prediction value match the expected ones
    assert(expected_output_classification == obtained_output_classification)
    assert(expected_output_regression == obtained_output_regression)
