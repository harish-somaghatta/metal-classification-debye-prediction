
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 16 Feb 2024
Description: Artificial Neural Network for binary classification.

"""
import numpy as np
import os
import sys
np.random.seed(10)
import matplotlib.pyplot as plt
pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))
sys.path.insert(0, pca_dir)

from optimization_algorithms import init_vel_momentum, init_adam, gradient_descent_with_momentum, gradient_descent_with_rmsprop, gradient_descent_with_adam, mini_batches
from classification_evaluation_metrics import accuracy, precision, recall, f1_score


def read_dataset(file):
    """
    Read the dataset from a CSV file and return shuffled independent and dependent features.

    Arguments:
    file (str): Path of the file containing the dataset.

    Returns:
    independent_features (numpy.ndarray): Shuffled independent features from the dataset (principal components).
    dependent_features (numpy.ndarray): Shuffled dependent features from the dataset (material_type and Debye temperature).
    """
    
    # Read all the features from csv file.
    all_features = np.genfromtxt(file, delimiter = ",", dtype = "float", skip_header = 1)
    #print(all_features)
    np.random.shuffle(all_features)
    # Get independent features
    independent_features = all_features[:, :-2]    
    # Get dependent features
    dependent_features = all_features[:,-2:]
    
    return independent_features, dependent_features


def split_dataset(independent_features, dependent_features, train_ratio):
    """
    Randomly splitting the dataset(Independent and Dependent features) into training set and test set.
    
    Arguments:
    independent_features (numpy.ndarray): Independent features from the dataset.
    dependent_feature (numpy.ndarray): Dependent feature from the dataset (material_type).
    train_ratio (float): Ratio of splitting the dataset into training and test sets.
    
    Returns:
    x_train (numpy.ndarray): Training set of independent features.
    x_test (numpy.ndarray): Testing set of independent features.
    y_train (numpy.ndarray): Training set of dependent features. 
    y_test (numpy.ndarray): Testing set of dependent features. 
    target_regression (numpy.ndarray): Randomly shuffled tagert feature for regression.
    """ 

    # Get the number of samples
    n_samples = independent_features.shape[0]
    dependent_feature_classification = dependent_features[:,0]
    dependent_feature_regression = dependent_features[:, -1].reshape(-1, 1)
    # Index of splitting the dataset based on training ratio
    split_idx = int(n_samples*train_ratio)
    
    # Split the dataset into training and test sets. 
    x_train, x_test = np.split(independent_features, [split_idx])
    y_train, y_test = np.split(dependent_feature_classification, [split_idx])
    # Get the randomly shuffled tagert feature for regression
    target_regression = dependent_feature_regression.reshape(-1, 1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    return x_train, x_test, y_train, y_test, target_regression


def initialize_parameters(num_neurons_list):
    """
    Initialize the weights and bias(parameters).
    
    Arguments:
    num_layers (list): List of number of neurons in each layer
    
    Returns:
    weights_list (list): List of weights for each layer.
    bias_list (list): List of bias for each layer.
    """
    # Initialize lists to store weights and biases of each layer.
    weights_list = [] 
    bias_list = [] 
    n_layers = len(num_neurons_list) - 1# Get the number of layers.
    #print(n_layers)
    
    # Iterate over the number of neurons.
    for i in range(n_layers):
        # Initialize non-zero weights matrix.
        weights = 0.01 * np.random.randn(num_neurons_list[i], num_neurons_list[i+1])
        weights_list.append(weights)
        # Initialize zero bias matrix.
        bias = np.zeros((1, num_neurons_list[i+1]))
        bias_list.append(bias)
        
    return weights_list, bias_list


def fwd_propagation(weights, bias, pre_activations):
    """
    Perform Linear forward propagation in the neural network.
    
    Arguments:
    weights (numpy.ndarray): Matrix of weights with size(previous_layer_size, current_layer_size)
    bias (numpy.ndarray): Vector of bias with size(1, current_layer_size)
    pre_activations (numpy.ndarray): Independent feature matrix or previous activation function
                                    with size(number_of_samples, number_of_features)
                                    
    Returns:
    lin_out (nu mpy.ndarray): Linear output of the forward propagation.
    
    """
    lin_out = np.dot(pre_activations, weights) + bias

    return lin_out

def relu_activation(linear_out):
    """
    The Rectified Linear Unit activation function is applied to every element.
    
    Arguments:
    linear_out (numpy.ndarray): The output of the linear transformation.
    
    Return:
    activation_res (numpy.ndarray): The result after application of Rectified Linear Unit activation.
    
    """
    # Get the maximum value of zero and linear transformation output for each element.
    activation_res = np.maximum(0., linear_out)
    
    return activation_res

def sigmoid_activation(linear_out):
    
    """
    The sigmoid activation function is applied to every element.
    
    Arguments:
    linear_out (numpy.ndarray): The output of the linear transformation.
    
    Return:
    sig_act (numpy.ndarray): The result after application of sigmoid activation.
    """

    # Get the sigmoid value for each element.
    sig_act = 1/(1+np.exp(-linear_out))
    
    return sig_act


def tanh_activation(linear_output):
    """
    The tanh activation function is applied to every element.
    
    Arguments:
    linear_out (numpy.ndarray): The output of the linear transformation.
    
    Return:
    tanh_act (numpy.ndarray): The result after application of tanh activation.
    
    """  
    tanh_act = np.tanh(linear_output)
    
    return tanh_act

def fwd_prop_model(x_train, p_weights, p_bias, activation_fn):
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
        
        # Get the activation matrix based on the activation matrix. 
        if activation_fn == "relu":
            activation = relu_activation(linear_output)
            activation_output.append(activation)
            #print(activation_output)
        elif activation_fn == "tanh":
            print("It is tanh")
            activation = tanh_activation(linear_output)
            activation_output.append(activation)
    
    # Linear forward propagation of the final layer.
    final_linear_output = fwd_propagation(p_weights[len_layers-1], p_bias[len_layers-1], activation)
    linear_output_list.append(final_linear_output)
    # Get the activation matrix from sigmoid activation function.
    last_activation = sigmoid_activation(final_linear_output)
    activation_output.append(last_activation)
    #print(last_activation)
    return last_activation, linear_output_list, activation_output


def cost_fn(last_activation, y_train):
    """
    Calculate the cross-entropy cost function for binary classification.
    
    Arguments:
    last_activation (numpy.ndarray): Activation matrix of the final layer.
    y_train (numpy.ndarray): The training dependent features(Metal(0) or Non-metal(1))

    Return:
    cost_val (float): The calculated value of the cross-entropy cost.
    
    """
    samples = y_train.shape[0] # Get the number of examples.
    # Compute cross-entropy cost function for binary classification based on last layer activation matrix.
    cost_val = (np.dot(np.log(last_activation).T,y_train) + np.dot(np.log(1-last_activation ).T,(1-y_train)))*(-1/samples)
    # Covert the cost value from numpy array to float value.
    cost_val = cost_val.item()

    return cost_val

def cost_with_reg(w, reg_par, last_activation, y_train):
    """
    Calculate the cross-entropy cost function for binary classification with regularization.
    
    Arguments:
    w (list): List containing weights of the neural network layers.
    reg_par (float): Regularization parameter.
    last_activation (numpy.ndarray): Activation matrix of the final layer.
    y_train (numpy.ndarray): The training dependent features(Metal(0) or Non-metal(1))

    Return:
    cost_reg (float): The calculated value of the regularized cross-entropy cost.
    
    """
    
    samples = y_train.shape[0] # Get the number of examples.
    
    # Compute cross-entropy cost function for binary classification based on last layer activation matrix.
    cost_val_ = (np.dot(np.log(last_activation).T,y_train) + np.dot(np.log(1-last_activation ).T,(1-y_train)))*(-1/samples)
    weights_sum = 0
    for weights in w:
        weights_sum += np.sum(weights**2)
    
    # Compute the regularization term.
    regularization_term = (reg_par/(2 * samples)) * weights_sum
    #print(regularization_term)
    # Covert the cost value from numpy array to float value.
    cost_val = cost_val_.item()
    # Calculate the regularized cost value.
    cost_reg = cost_val + regularization_term
    
    return cost_reg
    
    

def bwd_linear_prop(dj_d_lin, activation, weight, num_sam, reg_para):
    """
    Linear backward propagation
    
    Arguments:
    dj_d_lin (numpy.ndarray): Gradient of the cost w.r.t linear output.
    act (numpy.ndarray): Activation matrix.
    weight (numpy.ndarray): Weight matrix.
    num_sam (int): Number of samples.
    
    Returns:
    dj_dw (numpy.ndarray): Gradient of the cost w.r.t weight.
    dj_db (numpy.ndarray): Gradient of the cost w.r.t bias.
    old_dj_da (numpy.ndarray): Gradient of the cost w.r.t activation.
    
    """
    old_dj_da = np.dot(dj_d_lin, weight.T) 
    dj_dw = (np.dot(activation.T, dj_d_lin) + (weight * reg_para)) / num_sam
    # Get the mean of the linear output gradients along each column and keep the dimensions as it is.
    dj_db = np.mean(dj_d_lin, axis = 0, keepdims=True) 
    
    return dj_dw, dj_db, old_dj_da
    
    
    
def bwd_sigmoid(last_dj_da, activation):
    """
    Sigmoid activation function in the backward propagation.

    Arguments:
    last_dj_da (numpy.ndarray): Gradient of the cost w.r.t activation.
    activation (numpy.ndarray): Activation matrix.

    Returns:
    dj_d_lin (numpy.ndarray): Gradient of the cost w.r.t linear output.
    """
    # Peform element-wise multiplication.
    da_d_lin = activation * (1 - activation)
    dz = last_dj_da * da_d_lin

    return dz

def bwd_relu(old_dj_da, hidden_linear_output):
    """
    ReLU activation function of the backward propagation.

    Arguments:
    old_dj_da (numpy.ndarray): Gradient of the cost w.r.t activation.
    hidden_linear_output (numpy.ndarray): Linear output matrix.

    Returns:
    dj_d_lin_old (numpy.ndarray): Gradient of the cost w.r.t linear output.
    """
    # Perform element wise multiplication of gradient of the cost w.r.t activation with 1 for positive values in hidden_linear_output and 0 for non-positive values.
    dj_d_lin_old = old_dj_da * np.where(hidden_linear_output>0, 1, 0)
    
    return dj_d_lin_old
    

def bwd_prop_model(y, weights, bias, hidden_linear_output, activation_output, par_reg, activation_fn ):
    """
    Backward propagation model of the neural network for classification.

    Arguments:
    y (numpy.ndarray): Dependent feature matrix.
    weights (list): List of weight matrices of each layer.
    bias (list): List of bias matrices of each layer.
    hidden_linear_output (list): List of linear outputs from hidden layers.
    activation_output (list): List of activation outputs from all layers.
    activation_fn (str): Activation function to be used in hidden layers.

    Returns:
    d_weights (list): List of gradients of the w.r.t weight for each layer.
    d_bias (list): List of gradients of the cost w.r.t bias for each layer.
    """
    # Initialize lists to store gradient of cost function w.r.t activation, weight, bias, hidden linear output. 
    dA_list = []
    dW_list = []
    dB_list = []
    dZ_list = []
    
    num_layers = len(weights) # Get number of layers.
    #print(num_layers)
    n_samples = y.shape[0] # Get the number of samples.
    #print(n_samples)
    
    # Get the weights, activation of last layer.
    last_wei = weights[num_layers - 1] 
    last_act = activation_output[num_layers]
    #print(last_act)
    #print(y)
    # Compute gradient of cost w.r.t last layer activation.
    last_da = -(y/last_act) + ((1-y)/(1-last_act))
    #print(last_da.shape)
    dA_list.append(last_da)
    # Get the gradient of the cost w.r.t linear output.
    last_dz = bwd_sigmoid(last_da, last_act)  
    #print(last_dz.shape)
    #print((last_act - y))
    dZ_list.append(last_dz)
    # Get the gradients of cost w.r.t last layer weights, biases and last before layer activation.
    last_dw, last_db, old_da = bwd_linear_prop(last_dz, activation_output[num_layers-1], last_wei, n_samples, par_reg)
    dA_list.append(old_da)
    dW_list.append(last_dw) 
    dB_list.append(last_db)
    #print(f"shape of dA{num_layers}: {last_dj_da.shape}\nshape of dz{num_layers}: {last_dz.shape}\nshape of dw{num_layers}: {last_dj_dw.shape}\nshape of db{num_layers}: {last_dj_db.shape}\nshape of dA{num_layers-1}: {old_dj_da.shape}")
    
    # Iterate through the hidden layers in the reverse order.
    for i in range(num_layers-2, -1, -1):
        
        # Compute gradients of cost w.r.t hidden layer weights, biases for ReLU activation function.
        if activation_fn == "relu":
            old_dz = bwd_relu(dA_list[-1], hidden_linear_output[i])
            dZ_list.append(old_dz)
            old_dw, old_db, old_dA = bwd_linear_prop(dZ_list[-1], activation_output[i], weights[i], n_samples, par_reg)
            dA_list.append(old_dA)
            dW_list.append(old_dw)
            dB_list.append(old_db)
    dW_list = dW_list[::-1]
    dB_list = dB_list[::-1]

    return dW_list, dB_list
        

def parameter_update(para_weights, para_bias, dw, db, alpha):
    """
    Update the weights and biases of the neural network using gradient descent.

    Parameters:
    para_weights (list): List of weights of each layer.
    para_bias (list): List of biases of each layer.
    dw (list): List of gradients of the cost w.r.t weights.
    db (list): List of gradients of the cost w.r.t biases.
    alpha (float): Learning rate for gradient descent.

    Returns:
    para_weights (list): Updated weights.
    para_bias (list): Updated biases.
    """
    
    num_layers = len(dw) # Get the number of layers.

    # Iterate over each layer.
    for i in range(num_layers):
        # Update the weights and biases by gradient descent.
        para_weights[i] = para_weights[i] - alpha * dw[i]
        para_bias[i] = para_bias[i] - alpha * db[i]
        
    return para_weights, para_bias


def predict(x, updated_wt, updated_b):
    """
    Predicts the material type based on the updated weights and biases.

    Parameters:
    x (numpy array): Input features.
    updated_wt (numpy array): Updated weights.
    updated_b (numpy array): Updated biases.

    Returns:
    y_pre (numpy array): Predicts the material type (Non-metal: 1, metal: 0).
    
    """
    
    samples = x.shape[0]
    y_pre = np.zeros((samples, 1))
    last_activation, _, _ = fwd_prop_model(x, updated_wt, updated_b, activation_fn = "relu")
    # If the last activation value > 0.5, then it's non-metal, else it's a metal.
    y_pre = np.where(last_activation > 0.5, 1, 0)

    return y_pre

def learning_rate_decay(initial_alpha,decay_rate, epoch):
    """
    Compute the learning rate decay.

    Parameters:
    initial_alpha (float): Initial learning rate.
    decay_rate (float): Rate of decay.
    epoch (int): Number of echos.

    Returns:
    learning_rate_decay (float): Decay learning rate.
    
    """
    # Formula to compute learning rate decay
    learning_rate_decay = initial_alpha * (1/(1 + decay_rate * epoch))

    return learning_rate_decay

def evaluation_metrics_classification(predicted_labels, true_labels):
    try:
        accuracy_metric = accuracy(predicted_labels, true_labels)
        print("accuracy_metric: ", accuracy_metric)
        precision_metric = precision(predicted_labels, true_labels)
        print("precision: ", precision_metric)

        recall_metric = recall(predicted_labels, true_labels)
        print("recall: ", recall_metric)

        f1_score_metric = f1_score(precision_metric, recall_metric)
        print("f1_score: ", f1_score_metric)

    except ZeroDivisionError:
        accuracy_metric, precision_metric, recall_metric, f1_score_metric = 0, 0, 0, 0     
    
    return accuracy_metric, precision_metric, recall_metric, f1_score_metric
    

def write_metals_data(predict_training, predict_test, x_train, x_test, targetf_regression):
    """
    Write the metals data to a CSV file to use in regression tasks. Identifies the data points predicted as metals, and saves the corresponding features, predicted results, and target values to a CSV file.

    Parameters:
    predict_training (numpy.ndarray): Predicted results for the training data.
    predict_test (numpy.ndarray): Predicted results for the test data.
    x_train (numpy.ndarray): Features for the training data.
    x_test (numpy.ndarray): Features for the test data.
    targetf_regression (numpy.ndarray): Target values for the regression task.

    Returns:
    None
    """
    
    # Get the metals data, can be used for regression task.
    predicted_results = np.vstack((predict_training, predict_test))
    metal_indices = np.where(predicted_results == 1)[0]
    all_features_random = np.vstack((x_train, x_test))
    metals_data = np.column_stack((all_features_random[metal_indices], predicted_results[metal_indices], targetf_regression[metal_indices]))
    np.savetxt("input_file_regression.csv", metals_data, delimiter = ',', fmt = "%f")
    
    return None
    
def ANN_classification_main(x_train, x_test, y_train, y_test,epoch, initial_alpha, mini_batch_size, num_neurons, target_regression):
    """
    Main function in the neural network model.

    Parameters:
    epoch (int): Number of epochs for training.
    mini_batch_size (int): Size of mini-batches for mini-batch gradient descent.
    num_neurons (list): List of the number of neurons in each hidden layer.

    Returns:
    tuple: A tuple containing the final cost with regularization and various evaluation metrics for binary classification.
    """
    
    # Create mini-batches
    x_train_m, y_train_m = mini_batches(x_train, y_train, mini_batch_size)

    # List of number of layers in the network.
    layers = [x_train.shape[1]] + num_neurons  + [1]
    #print("Layers \n\n\n", layers)
    
    # Initialize parameters for the neural network.
    para_weights, para_bias = initialize_parameters(layers)
    # Set regularization parameter.
    reg_par = 0.8
    # Initialize velocity for momentum.
    vel_dw, vel_db = init_vel_momentum(para_weights, para_bias)
    # Initialize velocity and squared gradient for RMSprop.
    s_dw, s_db = init_vel_momentum(para_weights, para_bias)
    # Initialize velocity and squared gradient parameters for Adam optimization.
    v_dw, v_db, sr_dw, sr_db = init_adam(para_weights, para_bias)

    initial_alpha_gd = 0.2008810707215905
    #initial_alpha_gd = initial_alpha
    decay_rate, t = 0.1, 1
    costs = [] # List to store costs for plotting.
    
    # Iterate over given epoch
    for  i in range(epoch):
        
        # Iterate over mini-batches of training features.
        for x_mini, y_mini in zip(x_train_m, y_train_m):
            mom_para = 0.99
            para_rms = 0.9
            epsilon= 10e-8
            t = t + i
            
            # Compute learning rate decay
            #decay_alpha_gd = learning_rate_decay(initial_alpha_gd, decay_rate, i)
            decay_alpha_adam_rmsp = learning_rate_decay(initial_alpha, decay_rate, i)
            #print(decay_alpha)
            
            # Forward propagation
            last_activation, linear_output_hidden, activation_output = fwd_prop_model(x_mini, para_weights, para_bias, activation_fn = "relu")
            
            # Compute cost with regularization
            cost_with_regularization = cost_with_reg(para_weights, reg_par, last_activation, y_mini)
            
            # Backward propagation
            dj_dw,dj_db = bwd_prop_model(y_mini, para_weights, para_bias, linear_output_hidden,activation_output, reg_par, activation_fn = "relu")
            
            # Update parameters using stochastic gradient descent 
            #updated_weight, updated_bias = parameter_update(para_weights, para_bias, dj_dw, dj_db, decay_alpha_gd )
            
            #Update parameters using stochastic gradient descent with momentum.
            #_, _, updated_weight, updated_bias = gradient_descent_with_momentum(vel_dw, vel_db,para_weights, para_bias, dj_dw, dj_db, mom_para, decay_alpha_gd, t)
            
            # Update parameters using root mean square propagation.
            #_, _, updated_weight, updated_bias = gradient_descent_with_rmsprop(s_dw, s_db,para_weights, para_bias, dj_dw, dj_db, para_rms, epsilon, decay_alpha_rmsp)
            
            # Update parameters using Adam optimization.
            updated_weight, updated_bias = gradient_descent_with_adam(v_dw, v_db, sr_dw, sr_db, para_weights, para_bias, dj_dw, dj_db, mom_para, para_rms, decay_alpha_adam_rmsp, epsilon, t)
            
            if i%500 == 0:
                costs.append([i, cost_with_regularization])
                #print(i, cost_with_regularization)
    #plt.plot(costs)
    
    # Predict on training and test sets.
    predict_training = predict(x_train, updated_weight, updated_bias)
    predict_test = predict(x_test, updated_weight, updated_bias)
    
    if mini_batch_size >= 256 and target_regression is not None:
        # Write the metals data to csv filr, can be used for regression task.
        write_metals_data(predict_training, predict_test, x_train, x_test, target_regression)
    
    # Evaluation metrics
    print("Training evaluation metrics: ")
    train_accuracy, train_precision, train_recall, train_f1_score = evaluation_metrics_classification(predict_training, y_train)
    print("Testing evaluation metrics: ")
    test_accuracy, test_precision, test_recall, test_f1_score = evaluation_metrics_classification(predict_test, y_test)
     
    #cost_array = np.array(costs)
    
    #np.savetxt('cost_rms_.csv', cost_array, delimiter=',', header='iteration, cost_rmsprop')
    
    # Return regularized cost and evaluation metrics
    #return cost_with_regularization, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1_score, test_f1_score
    
    return predict_training, predict_test

if __name__ == "__main__":
    epoch = 15000
    m_batch = 256
    n_neurons = [30, 50]
    alpha = 0.000354728214309162
    file_path = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
    # Read dataset and return independent and dependent features.
    independent_features, dependent_feature = read_dataset(file_path)      
    # Split into train and test data.
    x_train, x_test, y_train, y_test, targetf_regression = split_dataset(independent_features, dependent_feature, train_ratio = 0.8 )
    predict_train, predict_test = ANN_classification_main(x_train, x_test, y_train, y_test,epoch, alpha, m_batch, n_neurons, targetf_regression)
    


