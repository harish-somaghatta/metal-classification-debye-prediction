"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 24 Mar 2024
Description: Unit test for ANN classification and regression models.

"""

# Import necessary libraries and functions
import numpy as np
import os
import sys
ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'ANN'))
sys.path.insert(0, ANN_dir)
from ANN_classification import split_dataset, initialize_parameters, fwd_propagation, relu_activation, sigmoid_activation, fwd_prop_model, cost_with_reg, bwd_linear_prop, bwd_sigmoid, bwd_relu, bwd_prop_model, parameter_update, predict, learning_rate_decay
from ANN_regression import linear_activation, fwd_prop_model_reg, mean_squared_error_with_regularization, bwd_prop_model_reg


#========================================================================================================

#Test function for verifying the functionality of splitting the dataset into train and test dataset.

def test_split_dataset():
    """
    
    # Purpose of test: Verify the functionality of split_dataset to split the dataset based on the specified train-test ratio.

    # Input: Numpy arrays of independent features, dependent features with train-test split ratio.
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Shape of the training and test splitted dataset should be splitted according to split ratio.

    # Obtained Output: The actual shapes of the obtained datasets after performing the split are compared with the expected shapes, which are same.
    
    """
    # Input
    independent_features = np.random.randn(10,4)
    dependent_features = np.random.randn(10,1)
    train_ratio = 0.8
    
    # Call the function to split the dataset
    obtained_x_train, obtained_x_test, obtained_y_train, obtained_y_test, obtained_target_regression = split_dataset(independent_features, dependent_features, train_ratio)
    
    # Expected shapes based on the train-test ratio
    expected_x_train_shape = (8, 4)
    expected_x_test_shape = (2, 4)
    expected_y_train_shape = (8, 1)
    expected_y_test_shape = (2, 1)
    expected_target_regression_shape = (10, 1)
    
    # Verification
    assert(expected_x_train_shape == obtained_x_train.shape)
    assert(expected_x_test_shape == obtained_x_test.shape)
    assert(expected_y_train_shape == obtained_y_train.shape)
    assert(expected_y_test_shape == obtained_y_test.shape)
    assert(expected_target_regression_shape == obtained_target_regression.shape)
    
#========================================================================================================

# Test function for verifying the random initialization of weights and zero bias.
def test_initialize_parameters():
    """
    # Purpose of test: Verify the functionality of initialize_parameters to initialize the weights and bias for a neural network with a specific number of layers and neurons.

    # Input: x_train: Randomly generated independent training features represented by a NumPy array of shape (10, 4). 
    #       List of number of neurons in each layer: [4, 3, 5, 1]
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: A list of numpy arrays, each representing the weights matrix and bias matrix for a layer of the neural network. The shape of each weight matrix and bias vector should match the number of neurons in the current and next layers.

    # Obtained Output: The shapes of these weight matrices are (4, 3), (3, 5), and (5, 1)
    #                  The shapes of these bias vectors are (1, 3), (1, 5), and (1, 1) 
    """
    # Randomly generated independent training features
    x_train = np.random.randn(10, 4)
    num_neurons_list = [x_train.shape[1], 3, 5, 1]
    
    # Call the function to initialize the parameters
    obtained_weight, obtained_bias = initialize_parameters(num_neurons_list)
    
    # Expected shapes of parameters
    expected_weight_shape_list = [(4, 3), (3, 5), (5, 1)]
    expected_bias_shape_list = [(1, 3), (1, 5), (1, 1)]
    
    # Verification
    for ex_wt, obt_wt in zip(expected_weight_shape_list, obtained_weight):
        assert(ex_wt == obt_wt.shape)
    for ex_bias, obt_bias in zip(expected_bias_shape_list, obtained_bias):
        assert(ex_bias == obt_bias.shape)
    
#========================================================================================================

# Test function for verifying the linear forward propagation.
def test_fwd_propagation():
    """
    # Purpose of test: Verify the functionality of fwd_propagation to calculate the linear output of a neural network layer based on the input, weights, and biases.

    # Input:  weights matrix with shape (3, 2), the bias vector with shape (2, 1),  pre-activation values with shape (2, 3).
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Matrix multiplication of the weights with the pre-activations matrix, then added by the bias matrix.  The expected linear output with shape (2, 2)

    # Obtained Output: The linear output obtained by forward propagation shape is (2, 2).

    """
    # Input 
    weights = np.array([[3, 2], [4, 5], [6, 2]])
    bias = np.array([[3], [4]])
    pre_activations = np.array([[1, 4, 3], [9, 5, 2]])
    # Calling the function
    obtained_linear_output = fwd_propagation(weights, bias, pre_activations)
    # Exprected results
    expected_linear_output = np.array([[40, 31], [63, 51]])
    expected_linear_output_shape = (2, 2)
    # Verification
    assert(expected_linear_output_shape == obtained_linear_output.shape)
    assert(obtained_linear_output == expected_linear_output).all()
    
#========================================================================================================

# Test function for verifying the Rectified Linear Unit activation function, max(0, linear_output(z)).
def test_relu_activation():
    """
    # Purpose of test: Verify the functionality of relu_activation, which introduces non-linearity in the neural network. It replaces negative linear output values with zero.

    # Input: linear output is np.array([[1, -2, 4, -5.5], [2, -4.2, -6.3, 0.9]])
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: ReLU activation should replace negative values with zero in the linear output and ignore positive values.

    # Obtained Output: np.array([[1, 0, 4, 0], [2, 0, 0, 0.9]])
    """
    # Input
    linear_out = np.array([[1, -2, 4, -5.5], [2, -4.2, -6.3, 0.9]])
    # Calling the function
    obtained_relu_output = relu_activation(linear_out)
    # Expected output
    expected_relu_output = np.array([[1, 0, 4, 0], [2, 0, 0, 0.9]])
    # Verification
    assert(linear_out.shape == obtained_relu_output.shape)
    assert(obtained_relu_output == expected_relu_output).all()

#========================================================================================================

# Test function for verifying the Sigmoid activation function ranges inbetween zero and one.
def test_sigmoid_activation():
    """
    # Purpose of test: Verify the functionality of sigmoid_activation, which introduces non-linearity in the neural network. It transform the values in the linear output matrix into the range (0, 1).

    # Input: np.array([[4, -1.5, 0.98], [0.64, 4.2, -3]])
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Sigmoid activation should transform the values in the linear output matrix into the range (0, 1). The sigmoid function is applied element wise.

    # Obtained Output: np.array([[0.98201379, 0.1824255238, 0.72710822], [0.65475346, 0.98522597, 0.04742587]])
	
    """
    linear_out = np.array([[4, -1.5, 0.98], [0.64, 4.2, -3]])
    # Calling the function
    obtained_sigmoid_output = sigmoid_activation(linear_out)
    # Expected output
    expected_sigmoid_output = np.array([[0.98201379, 0.1824255238, 0.72710822], [0.65475346, 0.98522597, 0.04742587]])
    # Verification
    assert(linear_out.shape == obtained_sigmoid_output.shape)
    assert(obtained_sigmoid_output - expected_sigmoid_output < 1e-3).all()
    
#========================================================================================================


# Test function for verifying the linear activation function of the model.
def test_linear_activation_regression():
    """
    # Purpose of test: Verify the functionality of linear_activation function, whether the function computes returns the linear output correctly.

    # Input: np.array([[1, 2, 6], [6, 0.5, 3.5]])
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Linear activation should give the same values as in the linear output before activation function.

    # Obtained Output: np.array([[1, 2, 6], [6, 0.5, 3.5]])
	
    """
    linear_output = np.array([[1, 2, 6], [6, 0.5, 3.5]])
    
    obtained_linear_activation = linear_activation(linear_output)
    
    expected_linear_activation = np.array([[1, 2, 6], [6, 0.5, 3.5]])
    
    assert(obtained_linear_activation == expected_linear_activation).all()
    
    
#========================================================================================================
# Test function for verifying the forward propagation of the model.

def test_fwd_prop_model():
    """
    # Purpose of test: Verify the functionality of fwd_prop_model function. This function provides the initial prediction, linear output and activation function output information.

    # Input: Numpy arrays of independent training features, weights, bias and the activation function of hidden layers.
			x_train = np.array([[0.5, 0.12], [1.5, -2.1]])
			p_weights = [np.array([[0.55, 0.1], [-0.4, 6.6]]), np.array([[0.425], [0.75]])]
			p_bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
			activation_fn = 'relu'

    # Expected Output: Compute the last activation, linear output and activation output using a simple numpy arrays.
                        last_activation =  [[0.84453952], [0.86524449]]
                    						linear_output = [np.array([[1.008, 0.592], [2.446, -13.96 ]]), np.array([[1.6924 ],[1.85955]])]
                    						activation_output = [np.array([[0.5 , 0.12], [1.5 , -2.1]]), np.array([[1.008, 0.592], [2.446, 0]]), np.array([[0.84453952], [0.86524449]])]
    # Command to run file: pytest test_ANN_all_functions.py

    # Obtained Output: 	last_activation =  [[0.84453952], [0.86524449]]
						linear_output = [np.array([[1.008, 0.592], [2.446, -13.96 ]]), np.array([[1.6924 ],[1.85955]])]
						activation_output = [np.array([[0.5 , 0.12], [1.5 , -2.1]]), np.array([[1.008, 0.592], [2.446, 0]]), np.array([[0.84453952], [0.86524449]])]
    """
    # Input
    x_train = np.array([[0.5, 0.12], [1.5, -2.1]])
    p_weights = [np.array([[0.55, 0.1], [-0.4, 6.6]]), np.array([[0.425], [0.75]])]
    p_bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
    activation_fn = 'relu'
    
    # Calling the function
    obtained_last_activation, obtained_linear_output_list, obtained_activation_output = fwd_prop_model(x_train, p_weights, p_bias, activation_fn)
    # Output lists are converted into Numpy arrays for consistency.
    
    expected_last_activation =  [[0.84453952], [0.86524449]]
    expected_linear_output = [np.array([[1.008, 0.592], [2.446, -13.96 ]]), np.array([[1.6924 ],[1.85955]])]
    expected_activation_output = [np.array([[0.5 , 0.12], [1.5 , -2.1]]), np.array([[1.008, 0.592], [2.446, 0]]), np.array([[0.84453952], [0.86524449]])]
    
    for exp_last_act, obt_last_act in zip(expected_last_activation, obtained_last_activation):
        assert(exp_last_act - obt_last_act< 1e-3).all()
    for exp_lin_out, obt_lin_out in zip(expected_linear_output, obtained_linear_output_list):
        assert(exp_lin_out - obt_lin_out< 1e-3).all()
    for exp_act_out, obt_act_out in zip(expected_activation_output, obtained_activation_output):
        assert(exp_act_out - obt_act_out< 1e-3).all()
#========================================================================================================
# Test function for verifying the functionality of fwd_prop_model_reg function. This function provides the initial prediction, linear output and activation function output information.
def test_fwd_prop_model_regression():
    """
    # Purpose of test: Verify the functionality of fwd_prop_model_reg function, which computes the last activation, linear output, and activation output using numpy arrays.

    # Input: Numpy arrays of independent training features, weights, bias, and the activation function of hidden layers.
            x_train = np.array([[1, 2], [0.5, 4]])
            p_weights = [np.array([[0.2, 0.3], [0.6, 0.95]]), np.array([[0.42], [0.12]])]
            p_bias = [np.array([[0.1, 0.5]]), np.array([[0.4]])]
            activation_fn = "relu"
            
    # Command to run file: pytest test_ANN_all_functions.py

    # Expected Output: Compute the last activation, linear output, and activation output using simple numpy arrays.
                        expected_last_activation = [[1.354], [2.026]]
                        expected_linear_output = [np.array([[1.5, 2.7], [2.6, 4.45]]), np.array([[1.354], [2.026]])]
                        expected_activation_output = [np.array([[1 , 2],[0.5, 4. ]]), np.array([[1.5, 2.7], [2.6 , 4.45]]), np.array([[1.354], [2.026]])]
    # Obtained Output: The last activation, linear output, and activation output are obtained after calling the fwd_prop_model function. The obtained output should match the expected output within a small tolerance, and the list lengths of the expected output and obtained output should also be the same.
                        last_activation = [[1.354], [2.026]]
                        linear_output = [np.array([[1.5, 2.7], [2.6, 4.45]]), np.array([[1.354], [2.026]])]
                        activation_output = [np.array([[1 , 2],[0.5, 4. ]]), np.array([[1.5, 2.7], [2.6 , 4.45]]), np.array([[1.354], [2.026]])]
    """
    # Input
    x_train = np.array([[1, 2], [0.5, 4]])
    p_weights = [np.array([[0.2, 0.3], [0.6, 0.95]]), np.array([[0.42], [0.12]])]
    p_bias = [np.array([[0.1, 0.5]]), np.array([[0.4]])]
    activation_fn = "relu"
        
    # Calling the function
    obtained_last_activation, obtained_linear_output_list, obtained_activation_output = fwd_prop_model_reg(x_train, p_weights, p_bias, activation_fn)
    
    # Expected output
    expected_last_activation = [[1.354], [2.026]]
    expected_linear_output = [np.array([[1.5, 2.7], [2.6, 4.45]]), np.array([[1.354], [2.026]])]
    expected_activation_output = [np.array([[1 , 2],[0.5, 4. ]]), np.array([[1.5, 2.7], [2.6 , 4.45]]), np.array([[1.354], [2.026]])]
    
    # Verification
    for exp_last_act, obt_last_act in zip(expected_last_activation, obtained_last_activation):
        assert np.allclose(exp_last_act, obt_last_act, atol=1e-3)
    for exp_lin_out, obt_lin_out in zip(expected_linear_output, obtained_linear_output_list):
        assert np.allclose(exp_lin_out, obt_lin_out, atol=1e-3)
    for exp_act_out, obt_act_out in zip(expected_activation_output, obtained_activation_output):
        assert np.allclose(exp_act_out, obt_act_out, atol=1e-3)
#========================================================================================================

# Test function for verifying the cost value with regularization.
def test_cost_with_reg():
    """
    # Purpose of test: Verify the functionality of cost_with_reg function which gives information about the error between predicted label and true label including regularization.

    # Input: Numpy arrays of weights, prediction label, true label and regularization parameter. 
            w = [np.array([[0.5, 0.45], [0.9, -1.6]]), np.array([[0.25, -0.5], [0.125, 0.62]])]
            reg_par = 0.1
            last_activation = np.array([[0.4], [0.8]])
            y_train = np.array([[0], [1]])
    # Expected Output: Compute the cost value after including the regularization term.
                        expected_cost = 0.48
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Obtained Output: The cost value returned by the function cost_with_reg is the obtained cost. If the expected cost value closely matches with the obtained cost, then the error is calculated correctly.
                        cost = 0.48
    
    """
    # Input
    w = [np.array([[0.5, 0.45], [0.9, -1.6]]), np.array([[0.25, -0.5], [0.125, 0.62]])]
    reg_par = 0.1
    last_activation = np.array([[0.4], [0.8]])
    y_train = np.array([[0], [1]])
    # Calling the function
    obtained_cost = cost_with_reg(w, reg_par, last_activation, y_train)
    # Expected cost
    expected_cost = 0.48
    # Verification
    assert(expected_cost - obtained_cost < 1e-3)

#========================================================================================================

def test_mean_squared_error_regression():
    """
    
    # Purpose of test: Verify the functionality of mean_squared_error_with_regularization function which gives information about the error between predicted label and true label including regularization.

    # Input: Numpy arrays of weights, prediction label, true label and regularization parameter. 
            weights = [np.array([[0.2, 0.3], [0.6, 0.95]]), np.array([[0.42], [0.12]])]
            y_true = np.array([[1.25], [2.5], [7.3]])
            y_pred = np.array([[1.05], [2.21], [7.2]])
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    
    # Expected Output: Compute the cost value after including the regularization term.
            expected_mse = 0.203
    # Obtained Output: The cost value returned by the function mean_squared_error_with_regularization is the obtained cost. If the expected cost value closely matches with the obtained cost, then the error is calculated correctly.
            mse = 0.203

    """
    weights = [np.array([[0.2, 0.3], [0.6, 0.95]]), np.array([[0.42], [0.12]])]
    y_true = np.array([[1.25], [2.5], [7.3]])
    y_pred = np.array([[1.05], [2.21], [7.2]])

    obtained_mse = mean_squared_error_with_regularization(weights, y_true, y_pred, reg_par = 0.6)
    expected_mse = 0.203
    assert(expected_mse - obtained_mse < 1e-3)
#========================================================================================================

# Test function for verifying the backward linear propagation.
def test_bwd_linear_prop():
    """
    # Purpose of test: Verify the functionality of bwd_linear_prop function (Backward linear propagation), which computes the gradient of loss w.r.t weights, bias, and activations.

    # Input: Numpy arrays of gradient of loss w.r.t linear output of the layer, activation output, and weights. Regularization parameter and number of samples in the dataset. 
            dj_d_lin = np.array([[0.1, 0.2], [0.3, 0.4]])
            activation = np.array([[0.5, 0.6], [0.7, 0.8]])
            weight = np.array([[1, 2], [3, 4]])
            reg_para = 0.1
            num_sam = weight.shape[0]
            
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the gradient of loss w.r.t weights, gradient of loss w.r.t bias, and gradient of loss w.r.t previous activation layer.
                    expected_dj_dw = np.array([[0.18, 0.29], [0.3, 0.42]])
                    expected_dj_db = np.array([[0.2, 0.3]])
                    expected_old_dj_da = np.array([[0.5, 1.1], [1.1, 2.5]])
    # Obtained Output: The gradient of loss w.r.t weights, gradient of loss w.r.t bias, and gradient of loss w.r.t previous activation layer obtained from the function bwd_linear_prop. The shape of the gradients of weight and activation layer should match with the weights and activation layer respectively.
    #                   Obtained gradients should match closely with the expected gradients within the tolerance of 1e-3.
                    dj_dw = np.array([[0.18, 0.29], [0.3, 0.42]])
                    dj_db = np.array([[0.2, 0.3]])
                    old_dj_da = np.array([[0.5, 1.1], [1.1, 2.5]])
    
    """
    # Input
    dj_d_lin = np.array([[0.1, 0.2], [0.3, 0.4]])
    activation = np.array([[0.5, 0.6], [0.7, 0.8]])
    weight = np.array([[1, 2], [3, 4]])
    reg_para = 0.1
    num_sam = weight.shape[0]
    # Calling the function
    obtained_dj_dw, obtained_dj_db, obtained_old_dj_da = bwd_linear_prop(dj_d_lin, activation, weight, num_sam, reg_para)
    # Expected gradient numpy arrays
    expected_dj_dw = np.array([[0.18, 0.29], [0.3, 0.42]])
    expected_dj_db = np.array([[0.2, 0.3]])
    expected_old_dj_da = np.array([[0.5, 1.1], [1.1, 2.5]])
    # Verification
    assert(obtained_dj_dw.shape == weight.shape)
    assert(obtained_old_dj_da.shape == activation.shape)
    assert(expected_dj_dw - obtained_dj_dw < 1e-3).all()
    assert(expected_dj_db - obtained_dj_db < 1e-3).all()
    assert(expected_old_dj_da - obtained_old_dj_da < 1e-3).all()
    
#========================================================================================================

# Test function for verifying the backward sigmoid activation function.

def test_bwd_sigmoid():
    """
    # Purpose of test: Verify the functionality of bwd_sigmoid function, which computes the gradients of loss w.r.t linear output of sigmoid activation function.

    # Input: Numpy arrays of gradient of loss w.r.t activation of the current layer and activation output of the current layer.
            last_dj_da = np.array([[0.4, 0.5], [0.2, 0.8]])
            activation = np.array([[0.4, 0.7], [0.5, 0.2]])
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the gradient of loss w.r.t linear output of sigmoid activation function.
            expected_dz = np.array([[0.096, 0.105], [0.05,  0.128]])
    # Obtained Output: The gradient of loss w.r.t linear output of sigmoid activation function obtained from the function bwd_sigmoid. The obtained gradients should closely match with the expected gradients witnin a tolerance of 1e-3.
            dz = np.array([[0.096, 0.105], [0.05,  0.128]])
    """
    # Input
    last_dj_da = np.array([[0.4, 0.5], [0.2, 0.8]])
    activation = np.array([[0.4, 0.7], [0.5, 0.2]])
    # Calling the function
    obtained_dz = bwd_sigmoid(last_dj_da, activation)
    # Expected gradient of loss w.r.t linear output of sigmoid activation function
    expected_dz = np.array([[0.096, 0.105], [0.05,  0.128]])
    # Verification
    assert(expected_dz - obtained_dz < 1e-3).all()
    
#========================================================================================================

# Test function for verifying the backward ReLU(Rectified Linear Unit) activation function.
def test_bwd_relu():
    """
    
    # Purpose of test: Verify the functionality of bwd_relu function, which computes the gradient of the loss with respect to the linear output of the ReLU activation function.

    # Input: Numpy arrays of gradient of loss w.r.t activation of the current layer and linear output of the current layer before ReLU activation application.
            old_dj_da = np.array([[0.6, 0.3], [0.2, 0.6]])
            hidden_linear_output = np.array([[0.1, 0.5], [0.7, -0.9]])
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the gradient of loss w.r.t linear output of ReLU activation function.
            expected_dj_d_lin_old = np.array([[0.6, 0.3], [0.2,0 ]])
    # Obtained Output: The gradient of loss w.r.t linear output of ReLU activation obtained from bwd_relu function. The obtained gradients should closely match with the expected gradients witnin a tolerance of 1e-3. 
            dj_d_lin_old = np.array([[0.6, 0.3], [0.2,0 ]])
    """
    # Input 
    old_dj_da = np.array([[0.6, 0.3], [0.2, 0.6]])
    hidden_linear_output = np.array([[0.1, 0.5], [0.7, -0.9]])
    # Calling the function
    obtained_dj_d_lin_old = bwd_relu(old_dj_da, hidden_linear_output)
    # Expected gradient of loss w.r.t linear output of relu activation function
    expected_dj_d_lin_old = np.array([[0.6, 0.3], [0.2,0 ]])
    # Verification
    assert(expected_dj_d_lin_old - obtained_dj_d_lin_old < 1e-3).all()
    
#========================================================================================================

# Test function for verifying the backward propagation of the model.
def test_bwd_prop_model():
    """
    
    # Purpose of test: Verify the functionality of bwd_prop_model function, which computes the gradient of the loss with respect to the parameters i.e weights and bias.

    # Input: Numpy arrays of independent training data and target label, list of weight matrices, list of bias vectors and activation function.
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the gradient of loss w.r.t weights and biases of each layer.

    # Obtained Output: The gradient of loss w.r.t weights and biases of each layer obtained from bwd_prop_model function. The obtained gradients should closely match with the expected gradients witnin a tolerance of 1e-3 and
    #                   The shape of the parameters should match the shape of the derivatives of loss w.r.t parameters.

    """
    # Input
    x_train = np.array([[0.6, 0.42], [8.5, -3.1]])
    y = np.array([[1], [0]])
    p_weights = [np.array([[0.55, 0.1], [-0.4, 4.6]]), np.array([[0.425], [0.75]])]
    p_bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
    activation_fn = 'relu'
    _, linear_out, act_out = fwd_prop_model(x_train, p_weights, p_bias, "relu")
    obtained_dw, obtained_db = bwd_prop_model(y, p_weights, p_bias, linear_out, act_out, 0.8, activation_fn)
    
    expected_dw = [np.array([[ 1.971,  0.0233], [-0.808,  1.828]]), np.array([[3.399], [0.235]])]
    expected_db = [np.array([[ 0.191, -0.0277]]), np.array([[0.450]])]
    # Verify the shape of the parameters and the shape of derivatives of loss w.r.t parameters
    for w, d_w in zip(p_weights, obtained_dw):
        assert(w.shape == d_w.shape)
    for b, d_b in zip(p_bias, obtained_db):
        assert(b.shape == d_b.shape)
    # Verify the expected and obtained derivaties of loss w.r.t parameters
    for exp_dw, obt_dw in zip(expected_dw, obtained_dw):
        assert(exp_dw - obt_dw < 1e-3).all()
    for exp_db, obt_db in zip(expected_db, obtained_db):
        assert(exp_db - obt_db < 1e-3).all()
    
    
#========================================================================================================

# Test function for verifying the backward propagation of the regression model.
def test_bwd_prop_model_regression():
    """
    
    # Purpose of test: Verify the functionality of test_bwd_prop_model function, which computes the gradient of the loss with respect to the parameters i.e weights and bias.
                     x_train = np.array([[0.52, 0.26], [6.5, -3.1]])
                     y = np.array([[0.95],[1.25]])
                     weights = [np.array([[0.55, 0.1], [-0.4, 4.6]]), np.array([[0.425], [0.75]])]
                     bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
                     activation_fn = 'relu'
    # Input: Numpy arrays of independent training data and target label, list of weight matrices, list of bias vectors and activation function.
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the gradient of loss w.r.t weights and biases of each layer.
                    expected_dw = [np.array([[3.024, 0.240], [-1.386, 1.940]]), np.array([[6.116], [0.812]])]
                    expected_db = [np.array([[0.632, 0.385]]), np.array([[1.488]])]

    # Obtained Output: The gradient of loss w.r.t weights and biases of each layer obtained from bwd_prop_model function. The obtained gradients should closely match with the expected gradients witnin a tolerance of 1e-3 and
    #                   The shape of the parameters should match the shape of the derivatives of loss w.r.t parameters.
                        dw = [np.array([[3.024, 0.240], [-1.386, 1.940]]), np.array([[6.116], [0.81285973]])]
                        db = [np.array([[0.632, 0.385]]), np.array([[1.488]])]
    """
    x_train = np.array([[0.52, 0.26], [6.5, -3.1]])
    y = np.array([[0.95],[1.25]])
    weights = [np.array([[0.55, 0.1], [-0.4, 4.6]]), np.array([[0.425], [0.75]])]
    bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
    activation_fn = 'relu'
    _, linear_out, act_out = fwd_prop_model_reg(x_train, weights, bias, activation_fn)
    obtained_dw, obtained_db = bwd_prop_model_reg(y, weights, bias, linear_out, act_out, activation_fn, reg_para = 0.8)
    
    expected_dw = [np.array([[3.024, 0.240], [-1.386, 1.940]]), np.array([[6.116], [0.81285973]])]
    expected_db = [np.array([[0.632, 0.385]]), np.array([[1.488]])]
    # Verify the shape of the parameters and the shape of derivatives of loss w.r.t parameters
    for w, d_w in zip(weights, obtained_dw):
        assert(w.shape == d_w.shape)
    for b, d_b in zip(bias, obtained_db):
        assert(b.shape == d_b.shape)
    # Verify the expected and obtained derivaties of loss w.r.t parameters
    for exp_dw, obt_dw in zip(expected_dw, obtained_dw):
        assert(exp_dw - obt_dw < 1e-3).all()
    for exp_db, obt_db in zip(expected_db, obtained_db):
        assert(exp_db - obt_db < 1e-3).all()
#========================================================================================================

# Test function for verifying the parameter updation step.
def test_parameter_update():
    """
    # Purpose of test: Verify the functionality of parameter_update function, which updates the parameters (weights and biases) correctly.

    # Input: Lists of weight matrices, bias vectors, gradients of loss w.r.t weights and biases, and learning rate.
            para_weights = [np.array([[0.55, 0.1], [-0.4, 4.6]]), np.array([[0.425], [0.75]])]
            para_bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
            dw = [np.array([[ 1.971,  0.0233], [-0.808,  1.828]]), np.array([[3.399], [0.235]])]
            db = [np.array([[ 0.191, -0.0277]]), np.array([[0.450]])]
            alpha = 0.1
    
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Calculate the updated parameters based on the gradients of parameters and learning rate.
            expected_updated_w = [np.array([[0.352, 0.0976], [-0.319, 4.417]]), np.array([[0.0851], [0.726]])]
            expected_updated_b = [np.array([[0.761, -0.247]]), np.array([[0.775]])]
    # Obtained Output: List of the updated weights and biases of each layer obtained from parameter_update function. The obtained values should closely match with the expected values witnin a tolerance of 1e-3.
            w = [np.array([[0.352, 0.0976], [-0.319, 4.417]]), np.array([[0.0851], [0.726]])]
            b = [np.array([[0.761, -0.247]]), np.array([[0.775]])]
    """
    
    para_weights = [np.array([[0.55, 0.1], [-0.4, 4.6]]), np.array([[0.425], [0.75]])]
    para_bias = [np.array([[0.781, -0.25]]), np.array([[0.82]])]
    dw = [np.array([[ 1.971,  0.0233], [-0.808,  1.828]]), np.array([[3.399], [0.235]])]
    db = [np.array([[ 0.191, -0.0277]]), np.array([[0.450]])]
    alpha = 0.1
    obtained_updated_w, obtained_updated_b = parameter_update(para_weights, para_bias, dw, db, alpha)
    
    expected_updated_w = [np.array([[0.352, 0.0976], [-0.319, 4.417]]), np.array([[0.0851], [0.726]])]
    expected_updated_b = [np.array([[0.761, -0.247]]), np.array([[0.775]])]
    
    for exp_updw, obt_updw in zip(expected_updated_w, obtained_updated_w):
        assert(exp_updw - obt_updw < 1e-3).all()
    for exp_updb, obt_updb in zip(expected_updated_b, obtained_updated_b):
        assert(exp_updb - obt_updb < 1e-3).all()
    
#========================================================================================================

#  Verify the function for predicting the label(Metal or Non-metal).
def test_predict():
    """
    
    # Purpose of test: Verify the functionality of the predict function, which predicts the labels (metal or non-metal) based on the last layer's predicted value.          

    # Input: Numpy array of independent training data, lists of updated weights and biases for each layer in the network. 
            x = np.array([[0.6, 0.42], [8.5, -3.1]])
            updated_wt = [np.array([[ 0.3529 , 0.09767], [-0.3192 , 4.4172 ]]), np.array([[0.0851], [0.7265]])]
            updated_b =  [np.array([[ 0.7619 , -0.24723]]), np.array([[0.775]])]
            
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the list of expected predicted labels. If the predicted value is greater than 0.5, it is said to be non-metal, and if it is less than 0.5, then the material is a metal. 
            expected_predicted_class = [[1], [1]]
    # Obtained Output: List of predicted labels obtained from predict function. The obtained labels should closely match with the expected labels.
            predicted_class = [[1], [1]]
    """
    
    x = np.array([[0.6, 0.42], [8.5, -3.1]])
    updated_wt = [np.array([[ 0.3529 , 0.09767], [-0.3192 , 4.4172 ]]), np.array([[0.0851], [0.7265]])]
    updated_b =  [np.array([[ 0.7619 , -0.24723]]), np.array([[0.775]])]
    obtained_predict_class = predict(x, updated_wt, updated_b)
    
    expected_predicted_class = [[1], [1]]
    
    for exp_class, obt_class in zip(expected_predicted_class, obtained_predict_class):
        assert(exp_class == obt_class)

#========================================================================================================
# Test function for verifying the learning rate decay value.
def test_learning_rate_decay():
    """   
    # Purpose of test: Verify the functionality of the test_learning_rate_decay function, whether the function correctly computes the decay in learning rate.      

    # Input: Initial learning rate, decay rate(rate at which learning rate reduces), number of epochs.
            initial_alpha = 0.1
            decay_rate = 0.5
            epoch = 10
            
    # Command to run file: pytest test_ANN_all_functions.py
    
    # Expected Output: Compute the expected learning rate decay.
            expected_lr_decay = 0.016
    # Obtained Output: Learning rate decay value obtained from learning_rate_decay function. The obtained value should closely match with the expected value within a tolerance of 1e-3.
            lr_decay = 0.0166
    """
    
    initial_alpha = 0.1
    decay_rate = 0.5
    epoch = 10
    obtained_lr_decay = learning_rate_decay(initial_alpha, decay_rate, epoch)
    expected_lr_decay = 0.0166
    
    assert(expected_lr_decay - obtained_lr_decay < 1e-3)

#========================================================================================================

    
    