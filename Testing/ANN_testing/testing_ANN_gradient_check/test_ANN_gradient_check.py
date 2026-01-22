"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 24 Feb 2024
Description: Gradient check method to verify the backward propagation of ANN model.

"""
import numpy as np
np.random.seed(10)
import os
import sys

ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..','ANN'))
sys.path.insert(0, ANN_dir)
from ANN_classification import read_dataset, split_dataset, initialize_parameters, fwd_prop_model, cost_fn, bwd_prop_model

# Purpose of the test: To check the Backward propagation of ANN classification(Gradient checking)
# Expected output: Gradient approximation values and original gradient values should be approximately equal(difference ~ 0).
# Obtained output: 2.5369569936859338e-05



def arrays_to_list(w_dw, b_db):
    """
    Convert arrays of weights and biases to a flattened list.

    Parameters:
    w_dw (list): List of weight arrays.
    b_db (list): List of bias arrays.

    Returns:
    flatten_list (list): Flattened list containing weights and biases.
    
    """
    # List to store flattened list of parameters.
    flatten_list = []
    
    # Iterate over all the parameters
    for dw_arrays, db_arrays in zip(w_dw, b_db):
        for dw_items in list(dw_arrays.flatten()):
            flatten_list.append(dw_items)
        for db_items in list(db_arrays.flatten()):
            flatten_list.append(db_items)
    
    return flatten_list

def list_to_array(flatten_list, w, b):
    """
    Convert a flattened list to arrays of weights and biases.

    Parameters:
    flatten_list (list): Flattened list containing weights and biases.
    w (list): List of weight arrays.
    b (list): List of bias arrays.

    Returns:
    weights_list (list): List of weight arrays bulit from the flattened list.
    bias_list (list): List of bias arrays bulit from the flattened list.
    
    """
    # Initialize lists to store weights and biases
    weights_list = []
    bias_list = []
    # Initialize index value
    idx_val = 0
    
    # Iterate over the weight and bias
    for weights_arr, b_arr in zip(w, b):
        idx_cutoff = weights_arr.shape[0] * weights_arr.shape[1]
        #print("idx_cutoff", idx_cutoff)
        idx_cutoff += idx_val
        #print("idx_cutoff_2:", idx_cutoff)
        weights_list.append(np.array(flatten_list[idx_val:idx_cutoff]).reshape(weights_arr.shape))
        #print(weights_list)
        idx_val = idx_cutoff
        #print("idx_val", idx_val)
        idx_cutoff = (b_arr.shape[0] * b_arr.shape[1]) + idx_val
        #print("idx_cutoff_3", idx_cutoff)
        bias_list.append(np.array(flatten_list[idx_val:idx_cutoff]).reshape(b_arr.shape))
        idx_val = idx_cutoff
    
    return weights_list, bias_list


def test_bwd_prop():
    """
    # Purpose of the test: To check the Backward propagation of ANN classification(Gradient checking)
    # Input: principal_components_with_dependent_features.csv
            weights, bias = initialize_parameters([x.shape[1], 4, 1])
            last_activation, linear_output_hidden, activation_output = fwd_prop_model(x, weights, bias, activation_fn = "relu")
            par_reg = 0.8
            epsilon = 1e-4
    #  Command to run file: pytest test_ANN_gradient_check.py
    # Expected output: Gradient approximation values and original gradient values should be approximately equal(difference ~ 0).
    # Obtained output: less than 1e-4
    """
    
    pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'pca'))
    sys.path.insert(0, pca_dir)
    file_path = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
    independent_features, dependent_feature = read_dataset(file_path)
    #independent_features_r, dependent_feature_r = read_dataset_reg("principal_components_dependent_features.csv")

    x, _, y, _,_ = split_dataset(independent_features, dependent_feature, train_ratio = 0.8)
                
    weights, bias = initialize_parameters([x.shape[1], 4, 1])
        
    last_activation, linear_output_hidden, activation_output = fwd_prop_model(x, weights, bias, activation_fn = "relu")
    #print(last_activation.shape)

    #cost_value = cost_fn(last_activation, y_train)
    par_reg = 0.8
    epsilon = 1e-4

    dw,db = bwd_prop_model(y, weights, bias, linear_output_hidden , activation_output, par_reg, activation_fn = "relu")
    
    
    cost_plus = []
    cost_minus = []
    grand_apprx = []
    num_list = []
    grad_list = arrays_to_list(dw, db)
    #print(grad_list)
    para_list = arrays_to_list(weights, bias)
    #print(para_list)
    
    for i in range(len(para_list)):
        
        para_add = np.copy(para_list)
        #print(para_add[:5])
        para_add[i] = para_add[i] + epsilon
        #print(para_add[:5])
        weights_plus, bias_plus = list_to_array(para_add, weights, bias)
        y_pred_plus, _, _ =  fwd_prop_model(x, weights_plus, bias_plus, activation_fn = "relu")
        cost_p = cost_fn(y_pred_plus, y)
        #print(cost_p)
        cost_plus.append(cost_p)
        
        para_sub = np.copy(para_list)
        #print(para_sub[:5])
        para_sub[i] = para_sub[i] - epsilon
        #print(para_sub[:5])
        weights_minus, bias_minus = list_to_array(para_sub, weights, bias)
        y_pred_minus, _, _  =  fwd_prop_model(x, weights_minus, bias_minus, activation_fn = "relu")
        cost_m = cost_fn(y_pred_minus, y)
        #print(cost_m)
        cost_minus.append(cost_m)
        
        approx_gradient = (cost_p - cost_m) / (2*epsilon)
        #print(approx_gradient)
        grand_apprx.append(approx_gradient)
        
        num = grad_list[i] - approx_gradient
        num_list.append(num)
    num = np.linalg.norm(num_list)
    den = np.linalg.norm(grad_list) + np.linalg.norm(grand_apprx)
    ratio  = (num/den)
    
    assert(ratio < 1e-4)
    
#num = test_bwd_prop(para_weights, para_bias, dj_dw, dj_db, x_train, y_train, epsilon=1e-7)

#==========================================================================================================
