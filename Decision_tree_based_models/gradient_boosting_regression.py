"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 06 May 2024
Description: Gradient boosting algorithm for predicting the Debye temperature of metals.

"""

# Import necessary library
import numpy as np
import os
import sys

ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ANN'))
sys.path.insert(0, ANN_dir)

# Import required functions from the files
from ANN_regression import read_dataset_reg, split_dataset
from classification_and_regression_tree import decision_tree, predict_val
from regression_evaluation_metrics import root_mean_squared_error, mean_absolute_error, r_squared_error


def gradient_boosting_trees(x_train, y_train, num_trees, lr, f0, max_depth, min_samples):
    """
    By generating regression decision trees one after the other to train gradient boosting model.

    Parameters:
    x_train (numpy.ndarray): Independent features of the training dataset.
    y_train (numpy.ndarray): Target feature of the training dataset.
    num_trees (int): Number of trees to be used for the model.
    lr (float): Learning rate, the rate of scaling of each tree's contribution.
    f0 (float): Initial prediction.
    max_depth (int): Maximum depth of each decision tree.
    min_samples (int): Minimum number of samples needed to split a node.

    Returns:
    trees (list): Decision trees list to be used for the model.
    """
    # Initialize a list to store decision trees
    trees = []
    # Initial prediction
    fm = f0
    
    # Iterate over number of trees
    for n_tree in range(int(num_trees)):
        
        #print(n_tree)
        
        # Calculate the residual between the target variable and the current prediction
        residual = y_train - fm
        # Train a regression tree to predict the residuals
        reg_tree = decision_tree(x_train, residual, max_depth, "regression", min_samples)
        #print(reg_tree, "\n\n")
        trees.append(reg_tree) 
        # Using the learned tree, make predictions and update the prediction ensemble
        y_pred = predict_val(x_train, reg_tree) 
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(-1, 1)
        # Update the prediction ensemble using the learning rate to scale the prediction 
        fm = fm + lr * y_pred
    
    # Return the list of decision trees
    return trees

def predict_gb(x_test, lr, trees, f0):
    """
    Predicts debye temperature using the gradient boosting regression model.

    Parameters:
    x_test (numpy.ndarray): Independent features of the test dataset.
    lr (float): Learning rate used during training.
    trees (list): List of trained decision trees forming the ensemble.
    f0 (float): Initial prediction.

    Returns:
    y_hat (numpy.ndarray): Predicted target values.
    """
        
    y_hat = f0 # Initialize the prediction with the mean of training dataset target variable 
    
    # Iterate over trees in the ensemble
    for tree in trees:
        
        # Make predictions based on the current tree
        y_tree_pred = predict_val(x_test, tree) 
        
        # Convert predictions to numpy array and reshape to one dimensional array
        y_tree_pred = np.array(y_tree_pred)
        y_tree_pred = y_tree_pred.reshape(-1, 1)
        
        # Update the prediction ensemble using the learning rate to scale the prediction 
        y_hat = y_hat + lr * y_tree_pred    
        #print(y_hat)
    
    # Return the predicted values using gradient boosting model
    return y_hat

def evalution_metrics(true_values, prediction):
    """
    Compute evaluation metrics for GB regression model.

    Parameters:
    true_values (numpy array): Actual target values of the test dataset.
    prediction (numpy array): Predicted target values of the test dataset.

    Returns:
    None
    """
    
    # Evaluation metrics
    test_rmse = root_mean_squared_error(true_values, prediction)
    print("Test RMSE:", test_rmse)
    
    test_mae = mean_absolute_error(true_values, prediction)
    print("Test mae:", test_mae)
    
    test_rse = r_squared_error(true_values, prediction)
    print("Test rse:", test_rse)
    
    return test_rmse, test_mae, test_rse

def gradient_boosting_main(x_train, x_test, y_train, y_test, num_trees, max_depth, min_samples):
    """
    Main function to execute gradient boosting for regression.

    Parameters:
    file (str): Path to the CSV file containing the dataset.
    num_trees (int): Number of trees to be used in the gradient boosting model.
    max_depth (int): Maximum depth of each decision tree.
    min_samples (int): Minimum number of samples needed to split a node.

    Returns:
    y_prediction (numpy.ndarray): Predicted target values for the test dataset.
    
    """
    # Read and split the dataset
    
    #print(x_train, y_train, x_test, y_test)
    lr = 0.1    # Learning rate
    # Initialize the model with the mean of the target training values
    f0 = np.mean(y_train)   
    # Get the trained trees
    gb_trees = gradient_boosting_trees(x_train, y_train, num_trees, lr, f0, max_depth, min_samples)
   
    y_pred = f0
    # Predict the value based on the trained trees
    y_prediction = predict_gb(x_test, lr, gb_trees, y_pred)
    
    rmse_, mae_, rsa_ = evalution_metrics(y_test, y_prediction)
    
    return y_prediction, rmse_, mae_, rsa_
    
if __name__ == "__main__":
    file_path = os.path.join(ANN_dir, 'input_file_regression.csv')
    independent_features, dependent_feature = read_dataset_reg(file_path)
    x_train, x_test, y_train, y_test,_ = split_dataset(independent_features, dependent_feature, train_ratio = 0.8)
    num_trees, max_depth, min_samples = 20, 20, 4
    y_prediction, rmse, mae, rse = gradient_boosting_main(x_train, x_test, y_train, y_test, num_trees, max_depth, min_samples)
    
    
    