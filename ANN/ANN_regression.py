
"""
course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 12 Mar 2024
Description: Artificial Neural Network for regression task (debye temperature prediction).

"""

import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt

from ANN_classification import split_dataset, initialize_parameters, fwd_propagation, relu_activation, tanh_activation, bwd_linear_prop, parameter_update, bwd_relu, learning_rate_decay
from optimization_algorithms import init_vel_momentum, init_adam, gradient_descent_with_momentum, gradient_descent_with_rmsprop, gradient_descent_with_adam, mini_batches

from regression_evaluation_metrics import root_mean_squared_error, mean_absolute_error, r_squared_error

def read_dataset_reg(file):
    """
    Read the dataset from a CSV file.

    Arguments:
    file (str): Path of the file containing the dataset.

    Returns:
    independent_features (numpy.ndarray): Independent features from the dataset (principal components).
    dependent_feature (numpy.ndarray): Dependent feature from the dataset (material_type).
    """
    
    # Read all the features from csv file.
    all_features = np.genfromtxt(file, delimiter = ",", dtype = "float", skip_header = 1)
    # Get independent features
    independent_features = all_features[:, :-2]    
    # Get dependent feature
    dependent_feature = all_features[:,-1]
    #print(dependent_feature[:11])
    # Calculate the mean for each independent features.
    mean_col = np.mean(dependent_feature, axis=0)
    # Calculate the standard deviation for each independent features.
    std_dev_col = np.std(dependent_feature, axis=0)
    # Scale independent features using Z-score scaling method.
    scaled_dependent_feature = (dependent_feature - mean_col) / std_dev_col
    dependent_feature = scaled_dependent_feature.reshape((scaled_dependent_feature.shape[0], 1))

    return independent_features, dependent_feature
    
def linear_activation(linear_output):
    """
    Linear activation function.
    
    Arguments:
    linear_output (numpy.ndarray): The output of the linear transformation.
    
    Return:
    activation_res (numpy.ndarray): The same as the input, as it is a linear activation.
    """
    activation_res = linear_output
    
    return activation_res

def fwd_prop_model_reg(x_train, p_weights, p_bias, activation_fn):
    """
    Forward linear model in the neural network and return linear output and final activation values.
    
    Arguments:
    x_train (numpy.ndarray): The training input features.
    p_weights (numpy.ndarray): The weight matrices for each layer within a list.
    p_bias (numpy.ndarray): The bias matrices for each layer within a list.
    activation_fn (str): Activation function to be used in the hidden layers.
    
    Return:
    last_activation (numpy.ndarray): Activation matrix of final layer.
    linear_output (numpy.ndarray): List of linear outputs of each hidden layer.
    final_linear_output (numpy.ndarray): Linear output of final layer.
    """
    
    len_layers = len(p_bias)  # Get the layers number in the network.
    #print(len_layers)
    activation = x_train # First activation will be the input features.
    # Initialize lists to store linear tranformation output and activation matrix of each layer.
    linear_output_list = [] 
    activation_output = [] 
    activation_output.append(activation)
    
    # Iterate over hidden layers.
    for i in range(len_layers-1):
        
        pre_activation = activation
        # Linear forward propagation
        linear_output = fwd_propagation(p_weights[i], p_bias[i], pre_activation)
        linear_output_list.append(linear_output)
        #print(f"Z:  \n{linear_output[:10, :]}")
        
        # Get the activation matrix based on the activation matrix. 
        if activation_fn == "relu":
            activation = relu_activation(linear_output)
            activation_output.append(activation)
            #print("activation_output: \n", activation[:10, :])
        elif activation_fn == "tanh":
            print("It is tanh")
            activation = tanh_activation(linear_output)
            activation_output.append(activation)
    
    # Linear forward propagation of the final layer.
    final_linear_output = fwd_propagation(p_weights[len_layers-1], p_bias[len_layers-1], activation_output[-1])
    #print("Z2: \n", final_linear_output[:10, :])
    linear_output_list.append(final_linear_output)
    # Get the activation matrix from linear activation function.
    last_activation = linear_activation(final_linear_output)
    activation_output.append(last_activation)
    #print(last_activation[:10, :])
    return last_activation, linear_output_list, activation_output


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true and predicted values.

    Args:
    y_true (numpy array): target values.
    y_pred (numpy array): Predicted target values.

    Returns:
    mse (float): Mean squared error.
    """
    cost = (1/y_true.shape[0]) * (np.sum((y_pred - y_true)**2))
    cost = np.squeeze(cost) 

    return cost

def mean_squared_error_with_regularization(weights, y_true, y_pred, reg_par):
    """
    Calculate the mean squared error between true and predicted values.

    Args:
    y_true (numpy array): target values.
    y_pred (numpy array): Predicted target values.

    Returns:
    mse (float): Mean squared error.
    """
    samples = y_true.shape[0] # Get the number of examples.
    cost = (1/y_true.shape[0]) * (np.sum((y_pred - y_true)**2))
    cost = np.squeeze(cost) # Cost as a value
    weights_sum = 0
    for w in weights:
        weights_sum += np.sum(w**2)
    
    # Compute the regularization term.
    regularization_term = (reg_par/(2 * samples)) * weights_sum

    return cost + regularization_term


def bwd_prop_model_reg(y, weights, bias, hidden_linear_output, activation_output, activation_fn, reg_para):
    """
    Backward propagation model of the neural network for regression.

    Arguments:
    y (numpy.ndarray): Dependent feature matrix.
    weights (list): List of weight matrices of each layer.
    bias (list): List of bias matrices of each layer.
    hidden_linear_output (list): List of linear outputs from hidden layers.
    activation_output (list): List of activation outputs from all layers.

    Returns:
    d_weights (list): List of gradients of the w.r.t weight for each layer.
    d_bias (list): List of gradients of the cost w.r.t bias for each layer.
    """
    # Initialize lists to store gradients of cost function w.r.t. weights and biases
    dW_list = []
    dB_list = []
    dA_list = []
    dZ_list = []
    
    num_layers = len(weights)  # Get number of layers
    n_samples = y.shape[0]  # Get the number of samples
    
    # Compute gradient of the cost w.r.t. the output of the neural network
    last_da = (2/ n_samples) * (activation_output[-1] - y) 
    dA_list.append(last_da)
    
    # Initialize the gradient of the cost w.r.t. the activation of the previous layer
    last_dz = last_da
    dZ_list.append(last_dz)
    last_wei = weights[num_layers - 1] 
    
    last_dw, last_db, old_da = bwd_linear_prop(last_dz, activation_output[num_layers-1], last_wei, n_samples, reg_para)
    #print(f"shape of dA{num_layers}: {last_da.shape}\nshape of dz{num_layers}: {last_dz.shape}\nshape of dw{num_layers}: {last_dw.shape}\nshape of db{num_layers}: {last_db.shape}\nshape of dA{num_layers-1}: {old_da.shape}")
    
    dA_list.append(old_da)
    dW_list.append(last_dw) 
    dB_list.append(last_db)
    
    for i in range(num_layers-2, -1, -1):
        
        # Compute gradients of cost w.r.t hidden layer weights, biases for ReLU activation function.
        if activation_fn == "relu":
            old_dz = bwd_relu(dA_list[-1], hidden_linear_output[i])
            dZ_list.append(old_dz)
            old_dw, old_db, old_dA = bwd_linear_prop(dZ_list[-1], activation_output[i], weights[i], n_samples, reg_para)
            dA_list.append(old_dA)
            dW_list.append(old_dw)
            dB_list.append(old_db)
    dW_list = dW_list[::-1]
    dB_list = dB_list[::-1]

    return dW_list, dB_list

def evaluation_metrics(pred_val, true_val):
    """
    Compute evaluation metrics for regression.

    Parameters:
    pred_val (numpy.ndarray): Predicted target values.
    true_val (numpy.ndarray): True target values.

    Returns:
    rmse_metric (float): Root mean squared error of the predictions.
    mae_metric (float): Mean absolute error of the predictions.
    rse_metric (float): R-squared error of the predictions.
    
    """
    # Compute Root mean squared error
    rmse_metric = root_mean_squared_error(true_val, pred_val)
    print("rmse error: ", rmse_metric)
    # Compute root Mean absolute error
    mae_metric = mean_absolute_error(true_val, pred_val)
    print("mae error: ", mae_metric)
    # Compute R-squared error
    rse_metric = r_squared_error(true_val, pred_val)
    print("R squared error: ", rse_metric)
    
    # Return evaluation metrics
    return rmse_metric, mae_metric, rse_metric

def plot_cost_iterations(cost_val_list):
    
    # Plot the cost over iterations
    plt.plot(cost_val_list)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. epochs')
    plt.show()
    
    return None
    
    

def main_reg(x_train, x_test, y_train, y_test, epoch, initial_alpha_adam, mini_batch_size, num_neurons):
    
    
    x_train_m, y_train_m = mini_batches(x_train, y_train, mini_batch_size)
    
    layers = [x_train.shape[1]] + num_neurons  + [1]
    
    para_weights, para_bias = initialize_parameters(layers)
    
    reg_par = 0.6  
    
    # Initialize velocity for momentum.
    vel_dw, vel_db = init_vel_momentum(para_weights, para_bias)
    # Initialize velocity and squared gradient for RMSprop.
    s_dw, s_db = init_vel_momentum(para_weights, para_bias)
    # Initialize velocity and squared gradient parameters for Adam optimization.
    v_dw, v_db, sr_dw, sr_db = init_adam(para_weights, para_bias)
    
    #initial_alpha_gd = 0.8
    #initial_alpha_rmsp = 0.0018136802614247848
    initial_alpha_adam = 0.0018136802614247848
    decay_rate, t = 0.2, 1
    
    cost_list = []
    for i in range(15000):
        
        for x_mini, y_mini in zip(x_train_m, y_train_m):
            mom_para = 0.99
            para_rms = 0.9
            epsilon= 10e-8
            t = t + i
            
            # Compute learning rate decay
            #decay_alpha_gd = learning_rate_decay(initial_alpha_gd, decay_rate, i)
            #decay_alpha_rmsp = learning_rate_decay(initial_alpha_rmsp, decay_rate, i)
            decay_alpha_adam = learning_rate_decay(initial_alpha_adam, decay_rate, i)
            
            # Forward propagation
            last_activation, linear_output_hidden, activation_output = fwd_prop_model_reg(x_mini, para_weights, para_bias, activation_fn="relu")
            
            # Compute cost (MSE)
            #cost_val = mean_squared_error(y_train, last_activation)
            
            cost_with_reg = mean_squared_error_with_regularization(para_weights, y_mini, last_activation, reg_par)
            
            # Backward propagation
            dW, dB = bwd_prop_model_reg(y_mini, para_weights, para_bias, linear_output_hidden, activation_output, "relu", reg_par )
            
            # Update parameters using stochastic gradient descent 
            #updated_weight, updated_bias = parameter_update(para_weights, para_bias, dW, dB, decay_alpha_gd)
            
            #Update parameters using stochastic gradient descent with momentum.
            #_, _, updated_weight, updated_bias = gradient_descent_with_momentum(vel_dw, vel_db,para_weights, para_bias, dW, dB, mom_para, decay_alpha_gd, t)
            
            # Update parameters using root mean square propagation.
            #_, _, updated_weight, updated_bias = gradient_descent_with_rmsprop(s_dw, s_db,para_weights, para_bias, dW, dB, para_rms, epsilon, decay_alpha_rmsp)
            
            # Update parameters using Adam optimization.
            updated_weight, updated_bias = gradient_descent_with_adam(v_dw, v_db, sr_dw, sr_db, para_weights, para_bias, dW, dB, mom_para, para_rms, decay_alpha_adam, epsilon, t)

            # Log cost for visualization
            if i % 1000 == 0:
                cost_list.append(cost_with_reg)
                
                #print(f"Iteration {i}, Cost: {cost_with_reg}")
    
    #plot_cost_iterations(cost_list)
    
    
    # Predictions and evaluation
    predict_training = fwd_prop_model_reg(x_train, updated_weight, updated_bias, activation_fn="relu")[0]
    predict_test = fwd_prop_model_reg(x_test, updated_weight, updated_bias, activation_fn="relu")[0]
    
    
    train_rmse, train_mae, train_rse = evaluation_metrics(predict_training, y_train)
    #print(train_rmse, train_mae, train_rse)
    test_rmse, test_mae, test_rse = evaluation_metrics(predict_test, y_test)
    #print(test_rmse, test_mae, test_rse)
    
    return predict_training, predict_test

if __name__ == "__main__":
    epoch = 15000
    m_batch = 256
    n_neurons = [4]
    alpha = 0.00354728214309162
    
    independent_features, dependent_feature = read_dataset_reg("input_file_regression.csv")
            
    x_train, x_test, y_train, y_test, _ = split_dataset(independent_features, dependent_feature, train_ratio = 0.8)
    
    main_reg(x_train, x_test, y_train, y_test, epoch, alpha, m_batch, n_neurons)
