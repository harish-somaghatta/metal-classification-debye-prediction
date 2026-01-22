"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 01 Apr 2024
Description: K-nearest neighbours algorithm to classify materials into metals and non-metals.

"""

# Import necessary libraries and functions
import numpy as np
import os
import sys
pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))
sys.path.insert(0, pca_dir)
ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ANN'))
sys.path.insert(0, ANN_dir)
from classification_evaluation_metrics import precision, accuracy, recall, f1_score
from ANN_classification import read_dataset, split_dataset

def cal_euclidean_distance(train_point, test_point):
    """
    Calculates the two points' Euclidean distance.

    Args:
    train_point (numpy array): Coordinates of a point in the training set.
    test_point (numpy array): Coordinates of a point in the test set.

    Returns:
    distance (float): Euclidean distance between the given two points.
    
    """

    distance = np.sum((train_point - test_point) ** 2)
    distance = np.sqrt(distance)
    
    return distance

def knn(x_train, y_train, x_test, k):
    """
    Performs k-nearest neighbors classification.

    Args:
    x_train (numpy array): Independent features of training data .
    y_train (numpy array): Target feature of training data.
    x_test (numpy array): Independent features of test data.
    k (int): K nearest neighbours to be considered.

    Returns:
    test_pred (list): Predicting the class of the test data.
    """
    
    # Initialize a list to store the predicted class of the test data.
    test_pred = []
    for test_val in x_test:
        # Initialize a list to store Euclidean distances between each training point and the test point.
        eucledian_distance_list = []
        
        # Iterate over all the training points.
        for train_val in x_train:
            eucledian_distance = cal_euclidean_distance(train_val, test_val)
            eucledian_distance_list.append(eucledian_distance)
        #print(eucledian_distance_list)
            
        # Sort the distances and get the indices of the nearest neighbors.
        sort_distance_idx = np.argsort(eucledian_distance_list)
        knn_idx = sort_distance_idx[:k]
        #print(knn_idx)
        
        # Get the target classes of the k nearest neighbors.
        knn_targets = y_train[knn_idx]
        unq_labels, repeat_val = np.unique(knn_targets, return_counts= True)
        
        # Find the label that nearest neighbors that is repeated.
        repeated_label_idx = np.argmax(repeat_val)
        repeated_label = unq_labels[repeated_label_idx]
        
        # Append the predicted label.
        test_pred.append(repeated_label)
            
    return test_pred

def calculate_evaluation_metrics(prediction, true_label):
    """
    Compute evaluation metrics for classification.
    
    Parameters:
    prediction (numpy array): Predicted labels for test dataset.
    true_label (numpy array): True labels for test dataset.
    
    Returns:
    test_accuracy (float): Accuracy of the test dataset.
    test_precision (float): Precision of the test dataset.
    test_recall (float): Recall of the test dataset.
    test_f1_score (float): F1-score of the test dataset.
    """
    
    try:
        # Compute accuracy of the test dataset
        test_accuracy = accuracy(prediction, true_label)
        print("Test accuracy: ", test_accuracy)
        
        # Compute precision of the test dataset
        test_precision = precision(prediction, true_label)
        print("Test precision: ", test_precision)
        
        # # Compute recall of the test dataset
        test_recall = recall(prediction, true_label)
        print("Test recall: ", test_recall)
        
        ## Compute F1-score of the test dataset
        test_f1_score = f1_score(test_precision, test_recall)
        print("Test f1_score: ", test_f1_score)
    except ZeroDivisionError:
        test_accuracy, test_precision, test_recall, test_f1_score = 0, 0, 0, 0
    
    return test_accuracy, test_precision, test_recall, test_f1_score
    

def main_knn(x_train, x_test, y_train, y_test, k):
    """
    Main function to execute k-nearest neighbors algorithm for classification.

    Parameters:
    input_file (str): Path to the CSV file containing dataset.
    k (int): Number of neighbors.

    Returns:
    prediction (numpy array): Predicted labels for the test data.
    """
    
    # Get the independent and dependent features.
    

    # Predict the class of the test data.
    prediction = knn(x_train, y_train, x_test, k)
    
    #accuracy, precision, recall, f1_score = calculate_evaluation_metrics(prediction, y_test)
    
    return prediction, f1_score
    
if __name__ == "__main__":
    k = 3
    file_path = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
    independent_features, dependent_feature = read_dataset(file_path)
    # Split the independent and dependent features into train and test data.
    x_train, x_test, y_train, y_test, _ = split_dataset(independent_features, dependent_feature, train_ratio = 0.8 )
    prediction, _ = main_knn(x_train, x_test, y_train, y_test, k)
    
