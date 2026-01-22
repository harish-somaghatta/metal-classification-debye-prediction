"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 04 Apr 2024
Description: Perform unit tests for each function of K-nearest neighbours classification algorithm.

"""
import numpy as np
import os
import sys

knn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'knn'))
print(knn_dir)
sys.path.insert(0, knn_dir)
from knn_classification import cal_euclidean_distance, knn

#==============================================================================================================================

#Test the functiion cal_euclidean_distance to calculate the Euclidean distance between two points. Compares the expected Euclidean distance with the obtained distance.

def test_euclidean_distance():
    """
    # Purpose of test: Calculates euclidean distance between two points by using cal_euclidean_distance function .

    # Input: Two points in the two-dimensional space.
            train_point = np.array([2, 3])
            test_point = np.array([5, 7])
    
    Command to run file: pytest test_knn.py
    
    # Expected Output: The Euclidean distance between two points is 5.0.
            expected_distance = 5.0
    # Obtained Output: Distance given by cal_euclidean_distance function is 5.0.
            obtained_distance = 5.0
    """
    train_point = np.array([2, 3])
    test_point = np.array([5, 7])
    expected_distance = 5.0
    obtained_distance = cal_euclidean_distance(train_point, test_point)
    
    assert(expected_distance == obtained_distance)

#==============================================================================================================================


# Tests K-nearest neighbours classification algorithm. This test case determines if the training samples and the chosen number of neighbors (k) are used by the KNN algorithm to accurately assign labels to the test samples.

def test_knn():
    
    """
    # Purpose of test: Tests the test_knn function which perform K-nearest neighbours algorithm to classify the material type.

    # Input: Independent and dependent training features, Independent test feature, number of neighbours to consider for classification(k). 
            # Training samples
            x_train = np.array([[1, 2], [3, 4], [2, 6], [1, 3]])
            y_train = np.array([0, 0, 1, 1])
            # Test sample
            x_test = np.array([[2, 5], [3, 5]])
            k = 3   # K nearest neighbours
            
    Command to run file: pytest test_knn.py
    
    # Expected Output: To predict the class of the target sample by the knn function.
            expected_labels = [1, 1]
    # Obtained Output: The output obtained by the knn function to predict the target sample class.
            obtained_labels = [1, 1]
    """
    
    # Training samples
    x_train = np.array([[1, 2], [3, 4], [2, 6], [1, 3]])
    y_train = np.array([0, 0, 1, 1])
    # Test sample
    x_test = np.array([[2, 5], [3, 5]])
    k = 3   # K nearest neighbours
    expected_labels = [1, 1]
    obtained_labels = knn(x_train, y_train, x_test, k)
    
    assert(expected_labels == obtained_labels)

#==============================================================================================================================


