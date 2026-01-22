"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 10 May 2024
Description: Multiple Linear Regression algorithm for predicting the Debye temperature of metals.

"""
# Import necessary libraries and functions
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ANN'))
sys.path.insert(0, ANN_dir)

# Import the required function from the corresponding files
from ANN_regression import read_dataset_reg, split_dataset, mean_squared_error_with_regularization, fwd_propagation
from regression_evaluation_metrics import root_mean_squared_error, mean_absolute_error, r_squared_error

def parameters_initialization(num_features):
    """
    Initialize the weights and biases.

    Parameters:
    num_features (int): Number of independent features.

    Returns:
    w (numpy array): Initialized weights.
    b (float): Initialized bias.
    
    """
    # Initialize zero vector for weights and set as zero for biases
    w = np.zeros(num_features).reshape(-1, 1)
    b = 0
    
    # Return initialized weights and bias
    return w, b


def calculate_gradient(x_train, y_train, weights, bias):
    """
    Calculate the gradient of the cost function to the weights and bias.

    Parameters:
    x_train (numpy array): Independent features of the training data.
    y_train (numpy array): Dependent feature of the training data.
    weights (numpy array): Initialized weights.
    bias (float): Initialized bias.

    Returns:
    dW (numpy array): Gradient of weights.
    dB (float): Gradient of bias.
    """
    
    # Use forward propagation to obtain predictions
    prediction = fwd_propagation(weights, bias, x_train)
    # Get the number of samples
    n_samples = y_train.shape[0]
    # Calculate the error between predictions and training dataset target values
    error = prediction - y_train
    
    # Compute the gradient of weights and bias
    dw = np.dot(x_train.T, error) * (1/n_samples) 
    db = np.sum(error) / n_samples
    
    return dw, db


def gradient_descent(w, b, dW, dB, learning_rate):
    """
    Calcuate gradient descent optimization to update parameters.

    Parameters:
    w (numpy array): Initial weights.
    b (float): Initial bias.
    x (numpy array): Independent features .
    y (numpy array): Dependent features.
    learning_rate (float): Learning rate for gradient descent.
    n_iter (int): Number of iterations for gradient descent.

    Returns:
    w (numpy array): Updated weights.
    b (float): Updated bias.
    cost_list (list): List of costs during iterations.
    """

        
    #  Update weights and bias using gradients and learning rate
    w = w - learning_rate * dW
    b = b - learning_rate * dB
    
    return w, b

def evaluation_metrics_regression(predict_train, y_train, predict_test, y_test):
    """
    Compute evaluation metrics for regression.

    Parameters:
    predict_train (numpy array): Predicted values for training data.
    y_train (numpy array): True values for training data.
    predict_test (numpy array): Predicted values for test data.
    y_test (numpy array): True values for test data.

    Returns:
    train_rmse (float): Root mean square metric for train dataset.
    test_rmse (float): Root mean square metric for test dataset.
    train_mae (float): Mean absolute error  metric for train dataset.
    test_mae (float): Mean absolute error metric for test dataset.
    train_rse (float): R-squared error metric for train dataset.
    test_rse (float): R-squared error metric for test dataset.
        
    """
    
    # Evaluation metrics
    train_rmse = root_mean_squared_error(predict_train, y_train)
    #print("Training RMSE:", train_rmse)
    test_rmse = root_mean_squared_error(predict_test, y_test)
    #print("Test RMSE:", test_rmse)
    train_mae = mean_absolute_error(y_train, predict_train)
    test_mae = mean_absolute_error(y_test, predict_test)
    #print("Training mae:", train_mae)
    #print("Test mae:", test_mae)
    train_rse = r_squared_error(y_train, predict_train)
    test_rse = r_squared_error(y_test, predict_test)
    #print("Training rse:", train_rse)
    #print("Test rse:", test_rse)

    return train_rmse, test_rmse, train_mae, test_mae, train_rse, test_rse

def plot_cost_fn(list_cost):
    """
    Plots the cost function values over iterations.

    Parameters:
    list_cost (list): A list of cost values and corresponding iterations.

    Returns:
    None
    """
    # Plot the cost values against the iterations
    plt.plot(list_cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()
    
    return None

def mlr_main(x_train, x_test, y_train, y_test, iteration_val):
    """
    Main function for training and evaluating a Multiple Linear Regression (MLR) algorithm.

    Parameters:
    file (str): Path to the CSV file containing the dataset.
    iteration_val (int): Number of iterations for training the model.

    Returns:
    predict_train (float): Predicted value for training dataset.
    predict_test (float): Predicted value for test dataset.
    """
    # Initialize parameters
    weights, bias = parameters_initialization(x_train.shape[1])
    cost_list = []
    
    for i in range(iteration_val):
    
        prediction = fwd_propagation(weights, bias, x_train)
        # Compute cost with regularization
        cost = mean_squared_error_with_regularization(weights, y_train, prediction, reg_par = 0.1)
        # Compute gradient of loss w.r.t parameters(weight, bias)
        dW, dB = calculate_gradient(x_train, y_train, weights, bias)
        # Update the weights and bias
        weights, bias = gradient_descent(weights, bias, dW, dB, learning_rate = 0.0005)
        
        if i % 1000 == 0:
            
            cost_list.append(cost)
           
            #print(f"Iteration {i}: Cost {cost}")
    #plot_cost_fn(cost_list)
    
    # Get the predicted values for training and test datasets
    predict_train = fwd_propagation(weights, bias, x_train)
    predict_test = fwd_propagation(weights, bias, x_test)
    
    # Get the evaluation metrics
    train_rmse, test_rmse, train_mae, test_mae, train_rse, test_rse = evaluation_metrics_regression(predict_train, y_train, predict_test, y_test)
    
    return predict_train, predict_test
    
if __name__ == "__main__":
    file_path = os.path.join(ANN_dir, 'input_file_regression.csv')
    independent_features, dependent_feature = read_dataset_reg(file_path)
    x_train, x_test, y_train, y_test,_ = split_dataset(independent_features, dependent_feature, train_ratio = 0.8)
    iterations = 10000
    mlr_main(x_train, x_test, y_train, y_test, iterations)
    
    