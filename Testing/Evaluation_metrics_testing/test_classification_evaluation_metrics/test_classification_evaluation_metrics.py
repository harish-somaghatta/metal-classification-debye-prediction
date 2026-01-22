"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 16 Mar 2024
Description: Unit test for classification evaluation metrics.

"""
import numpy as np
import os
import sys
ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'ANN'))
sys.path.insert(0, ANN_dir)
from classification_evaluation_metrics import accuracy, precision, recall, f1_score

#===============================================================================================================

# Test function to verify the accuracy calculation.

def test_accuracy():
    """
    # Purpose of test: This test make sure that the function accuracy, calculates the accuracy betweeen the predictions and true labels correctly.

    # Input: True labels and predictions for the set of datapoints.
            true_label = np.array([1, 1, 1, 0, 1])
            prediction = np.array([1, 1, 0, 0, 1])
            
    Command to run file: pytest test_classification_evaluation_metrics.py
    
    # Expected output: The accuracy is calculated based on the predicted and true labels.
            expected_accuracy = 80
    # Obtained output: The accuracy obtained by calling the function accuracy, matched with the expected accuracy.
            obtained_accuracy = 8-
    """
    # True labels and predictions
    true_label = np.array([1, 1, 1, 0, 1])
    prediction = np.array([1, 1, 0, 0, 1])
    # Calculate accuracy by calling the function
    obtained_accuracy = accuracy(prediction, true_label)
    # Expected accuracy percentage
    expected_accuracy = 80
    
    # Check if the obtained accuracy matches the expected value
    assert(expected_accuracy == obtained_accuracy)

#===============================================================================================================

# Test function to verify the precision calculation.

def test_precision():
    """
    # Purpose of test: This test make sure that the function precision, calculates the precision betweeen the predictions and true labels correctly.

    # Input: True labels and predictions for the set of datapoints.
        true_label = np.array([1, 1, 1, 0, 1])
        prediction = np.array([1, 1, 0, 1, 1])
    
    Command to run file: pytest test_classification_evaluation_metrics.py
    
    # Expected output: The precision is calculated based on the predicted and true labels.
        expected_precision = 0.75
    # Obtained output: The precision obtained by calling the function precision, should match with the expected precision.
        obtained_precision = 0.75
    """
    # True labels and predictions
    true_label = np.array([1, 1, 1, 0, 1])
    prediction = np.array([1, 1, 0, 1, 1])
    # Calculate precision by calling the function
    obtained_precision = precision(prediction, true_label)
    # Expected precision value
    expected_precision = 0.75
    
    # Check if the obtained precision matches the expected value
    assert(expected_precision - obtained_precision < 1e-3)

#===============================================================================================================
# Test function to verify the recall calculation.
def test_recall():
    """
    
    # Purpose of test: This test make sure that the function recall, calculates the recall betweeen the predictions and true labels correctly.

    # Input: True labels and predictions for the set of datapoints.
        prediction = np.array([1, 1, 0, 1, 1])
        true_label = np.array([1, 1, 1, 0, 1])
    
    Command to run file: pytest test_classification_evaluation_metrics.py
    
    # Expected output: The recall is calculated based on the predicted and true labels.
        expected_recall = 0.75
    # Obtained output: The recall obtained by calling the function recall, should match with the expected recall.
        obtained_recall = 0.75
    """
    # True labels and predictions
    prediction = np.array([1, 1, 0, 1, 1])
    true_label = np.array([1, 1, 1, 0, 1])
    # Calculate recall by calling the function
    obtained_recall = recall(prediction, true_label)
    # Expected recall value
    expected_recall = 0.75
    
    # Check if the obtained recall matches the expected value
    assert(expected_recall - obtained_recall  < 1e-3)

#===============================================================================================================

# Test function to verify the f1 score calculation.
def test_f1_score():
    """
    
    # Purpose of test: This test make sure that the function f1_score, calculates the f1_score betweeen the predictions and true labels correctly.

    # Input: Precision and recall for a given class.
        precision = 0.75
        recall = 0.80
    
    Command to run file: pytest test_classification_evaluation_metrics.py
    
    # Expected output: The f1 score is calculated based on the Precision and recall values.
        expected_f1score = 0.774
    # Obtained output: The recall obtained by calling the function f1_score, should match with the expected recall.
        obtained_f1score = 0.774
    """
    # Precision and recall
    precision = 0.75
    recall = 0.80
    # Calculate f1 score by calling the function
    obtained_f1score = f1_score(precision, recall)
    # Expected f1 score value
    expected_f1score = 0.774
    
    # Check if the obtained recall matches the expected value
    assert(expected_f1score - obtained_f1score  < 1e-3)
    
#===============================================================================================================
    
    
    
