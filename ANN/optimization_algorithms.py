"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 20 Feb 2024
Description: Optimization algorithms used for Artificial Neural Network model.

"""
import numpy as np
np.random.seed(10)

def init_vel_momentum(w, b):
    """
    Initialize velocity for stochastic gradient descent with momentum optimization.

    Parameters:
    w (list): List of weights for each layer.
    b (list): List of bias for each layer.

    Returns:
    vel_dwlist (list): List containing veloctyif weights.
    vel_dblist (list): Lists containing velocity biases.
    """
    # Initialize lists to store velocities for weights and biases.
    vel_dwlist = []
    vel_dblist = []
    # Get the number of layers in the neural network.
    n_layers = len(w)
    
    # Iterate over each layer to initialize velocities for weights and biases.
    for i in range(n_layers):
        # Initialize velocity for weights with zeros of the current layer.
        vel_dw = np.zeros(w[i].shape)
        vel_dwlist.append(vel_dw)
        vel_db = np.zeros(b[i].shape)
        vel_dblist.append(vel_db)
    
    return vel_dwlist, vel_dblist

def init_adam(w, b):
    """
    Initialize velocities and squared gradients for Adam optimization.

    Parameters:
    w (list): List of weights for each layer.
    b (list): List of biases for each layer.

    Returns:
    vel_dwlist (list): List containing velocity of weights.
    vel_dblist (list): List containing velocity of biases.
    s_dwlist (list): List containing squared gradients of weights.
    s_dblist (list): List containing squared gradients of biases.
    
    """
    # Initialize lists to store velocities and squared gradients for weights and biases.
    vel_dwlist = []
    vel_dwlist = []
    vel_dblist = []
    s_dwlist = []
    s_dblist = []
    
    # Get the number of layers in the neural network.
    n_layers = len(w)
    
    # Iterate over each layer to initialize velocities and squared gradients for weights and biases.
    for i in range(n_layers):
        vel_dw = np.zeros(w[i].shape)
        vel_dwlist.append(vel_dw)
        vel_db = np.zeros(b[i].shape)
        vel_dblist.append(vel_db)
    
        s_dw = np.zeros(w[i].shape)
        s_dwlist.append(s_dw)
        s_db = np.zeros(b[i].shape)
        s_dblist.append(s_db)
    
    # Return the lists of velocities and squared gradients for weights and biases.
    return vel_dwlist, vel_dblist, s_dwlist, s_dblist
    

def gradient_descent_with_momentum(v_dw, v_db, w, b, dw, db, mom_para, alpha, t):
    """
    Update weights and biases using stochastic gradient descent with momentum optimization.

    Parameters:
    v_dw (list): List containing velocity of weights for each layer.
    v_db (list): List containing velocity of biases for each layer.
    w (list): List of weights for each layer.
    b (list): List of biases for each layer.
    dw (list): List containing gradients of weights for each layer.
    db (list): List containing gradients of biases for each layer.
    mom_para (float): Momentum parameter.
    alpha (float): Learning rate.
    t (int): Time step.

    Returns:
    v_dw (list): Updated list containing velocity of weights.
    v_db (list): Updated list containing velocity of biases.
    w (list): Updated list of weights.
    b (list): Updated list of biases.
    
    """
    # Get the number of layers.
    n_layers = len(dw)    
    vdw_corr = []
    vdb_corr = []
    
    # Iterate over each layer to update velocities, weights, and biases.
    for j in range(n_layers):
        # Update velocity for weights and bias using momentum.
        v_dw[j] = (mom_para * v_dw[j]) + ((1-mom_para) * dw[j])
        v_db[j] = (mom_para * v_db[j]) + ((1-mom_para) * db[j])
        
        # Correct velocity for weights and bias with the initial time steps.
        vdw_corrected = v_dw[j] / (1 - (mom_para ** t))
        vdw_corr.append(vdw_corrected)
        vdb_corrected = v_db[j] / (1 - (mom_para ** t))
        vdb_corr.append(vdb_corrected)
        
        # Update weight and bias using the corrected velocity.
        w[j] = w[j] - alpha * v_dw[j]
        b[j] = b[j] - alpha * v_db[j]
    
    # Return the updated lists of velocity and parameters.
    return v_dw, v_db, w, b

def gradient_descent_with_rmsprop(s_dw, s_db, w, b, dw, db, rms_para, eps, alpha):
    """
    Update weights and biases using RMSprop optimization.

    Parameters:
    s_dw (list): List containing squared gradients of weights for each layer.
    s_db (list): List containing squared gradients of biases for each layer.
    w (list): List of weights for each layer.
    b (list): List of biases for each layer.
    dw (list): List containing gradients of weights for each layer.
    db (list): List containing gradients of biases for each layer.
    rms_para (float): RMSprop parameter.
    eps (float): Minimal value to avoid zero division.
    alpha (float): Learning rate.

    Returns:
    s_dw (list): Updated list containing squared gradients of weights.
    s_db (list): Updated list containing squared gradients of biases.
    w (list): Updated list of weights.
    b (list): Updated list of biases.
    """
    
    # Get the number of layers.
    n_layers = len(dw)
    
    # Iterate over each layer.
    for k in range(n_layers):
        # Update squared gradients for weights and biases using RMSprop optimization.
        s_dw[k] = (rms_para * s_dw[k]) + ((1-rms_para) * (dw[k]**2))
        s_db[k] = (rms_para * s_db[k]) + ((1-rms_para) * (db[k]**2))
        
        # Update weight and bias using stochastic gradient descent with RMS prop optimization. 
        w[k] = w[k] - (alpha * (dw[k] / (np.sqrt(s_dw[k] + eps))))
        b[k] = b[k] - (alpha * (db[k] / (np.sqrt(s_db[k] + eps))))
    
    # Return the updated lists of squared gradients and parameters(weight and bias).
    return s_dw, s_db, w, b

def gradient_descent_with_adam(v_dw, v_db, s_dw, s_db, w, b, dw, db, mom_para, rms_para, alpha, eps, t):
    """
    Update weights and biases using Adam optimization.

    Parameters:
    v_dw (list): List containing velocity of gradients of weights for each layer.
    v_db (list): List containing velocity of gradients of biases for each layer.
    s_dw (list): List containing squared gradients of weights for each layer.
    s_db (list): List containing squared gradients of biases for each layer.
    w (list): List of weights for each layer.
    b (list): List of biases for each layer.
    dw (list): List containing gradients of weights for each layer.
    db (list): List containing gradients of biases for each layer.
    mom_para (float): Momentum parameter.
    rms_para (float): RMSprop parameter.
    alpha (float): Learning rate.
    eps (float): Minimal value to avoid zero division.

    Returns:
    w (list): Updated list of weights.
    b (list): Updated list of biases.
    
    """
    # Initialize lists to store corrected velocities and squared gradients.
    vdw_corr = []
    vdb_corr = []
    sdw_corr = []
    sdb_corr = []
    
    # Get the number of layers in the neural network.
    n_layers = len(dw)
    
    # Iterate over each layer to update velocities, squared gradients, weights, and biases.
    for i in range(n_layers):
        # Update velocity of gradients for weights and biases using momentum.
        v_dw[i] = (mom_para * v_dw[i]) + ((1-mom_para) * dw[i])
        v_db[i] = (mom_para * v_db[i]) + ((1-mom_para) * db[i])
        
        # Update the Corrected velocity of gradients for weights and biases.
        vdw_corrected = v_dw[i] / (1 - (mom_para ** t))
        vdw_corr.append(vdw_corrected)
        vdb_corrected = v_db[i] / (1 - (mom_para ** t))
        vdb_corr.append(vdb_corrected)
        
        # Update squared gradients for weights and biases using RMSprop.
        s_dw[i] = (rms_para * s_dw[i]) + ((1-rms_para) * (dw[i]**2))
        s_db[i] = (rms_para * s_db[i]) + ((1-rms_para) * (db[i]**2))
        
        # Update the corrected squared gradients for weights.
        sdw_corrected = s_dw[i] / (1 - (rms_para ** t))
        sdw_corr.append(sdw_corrected)
        sdb_corrected = s_db[i] / (1 - (rms_para ** t))
        sdb_corr.append(sdb_corrected)
        
        # Update the weight and bias using corrected velocity and corrected squared gradients.
        w[i] = w[i] - alpha * (vdw_corrected / (np.sqrt(sdw_corrected + eps)))
        b[i] = b[i] - alpha * (vdb_corrected / (np.sqrt(sdb_corrected + eps)))
    
    # Return updated weight and bias using adam optimization.
    return w, b
   
def mini_batches(x_train, y_train, mini_batch_size):
    """
    Create mini-batches for training data.

    Parameters:
    x_train (numpy.ndarray): Training set of independent features.
    y_train (numpy.ndarray): Training set of target feature.
    mini_batch_size (int): Size of each mini-batch. 

    Returns:
    x_train_mini_batch (list): List containing mini-batches of independent features.
    y_train_mini_batch (list): List containing mini-batches of target feature.
    
    """
    # Get the number of samples in the training data.
    m = x_train.shape[0]
    # Initialize lists to store mini-batches of independent features and target feature.
    x_train_mini_batch = []
    y_train_mini_batch = []
    # Create indices and shuffle the training data.
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_shuffled = x_train[indices, :]
    y_shuffled = y_train[indices, :]
    # Calculate the number of mini-batches.
    n_mini_batches = int(m/mini_batch_size)
    
    # Iterate over each mini-batch.
    for i in range(n_mini_batches):
        # Get mini-batches of independent features and target feature.
        x_mini_batch = x_shuffled[ (i * mini_batch_size) : (mini_batch_size * (i+1)), :]
        y_mini_batch = y_shuffled[ (i * mini_batch_size) : (mini_batch_size * (i+1)), :]
        
        x_train_mini_batch.append(x_mini_batch)
        y_train_mini_batch.append(y_mini_batch)
    
    # Compute the last mini batch size
    last_mini_batch = m - (mini_batch_size * n_mini_batches)
    # Get the last mini batch of independent features and target feature.
    x_train_last_mbatch = x_shuffled[-last_mini_batch:, :]
    x_train_mini_batch.append(x_train_last_mbatch)
    y_train_last_mbatch = y_shuffled[-last_mini_batch:, :]
    y_train_mini_batch.append(y_train_last_mbatch)
    
    # Return the list of training dataset mini batches.
    return x_train_mini_batch, y_train_mini_batch


        
        
    
    
    
    
    

