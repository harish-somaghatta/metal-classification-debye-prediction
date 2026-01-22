"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 25 Feb 2024
Description: Perform unit test of optimization algorithms used for ANN.

"""
import numpy as np
import os
import sys

ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'ANN'))
sys.path.insert(0, ANN_dir)
from optimization_algorithms import init_vel_momentum, init_adam, gradient_descent_with_momentum, gradient_descent_with_rmsprop, gradient_descent_with_adam, mini_batches

#========================================================================================================

# Test function for verifying the initialization of velocity for Stochastic gradient with momentum optimization.
def test_init_vel_momentum():
    """
    # Purpose of test: Verify the functionality of init_vel_momentum to initialize the velocity for the weights and biasas in the neural network.
        
    # Input: Numpy arrays of weights and biases of the neural network.
            w = [np.array([[0.5, 0.15], [-0.45, 6.6]]), np.array([[0.9], [0.5]])]
            b = [np.array([[0.8, -0.2]]), np.array([[0.82]])]
    # Expected Output: Numpy arrays list of initial velocity of weights and biases, initialized as zero arrays.
            expected_dw_vel_init = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
            expected_db_vel_init = [np.array([[0, 0]]), np.array([[0]])]
    
    Command to run file: pytest test_optimization_algorithms.py
    
    # Obtained Output: The actual numpy arrays list of initial velocity of weights and bias obtained from init_vel_momentum function have the same shape as weights and bias respectively.
    """
    
    # Initial simple array of weights and biases
    w = [np.array([[0.5, 0.15], [-0.45, 6.6]]), np.array([[0.9], [0.5]])]
    b = [np.array([[0.8, -0.2]]), np.array([[0.82]])]
    
    # Calling the function to initialize velocity for momentum
    obtained_dw_vel_init, obtained_db_vel_init = init_vel_momentum(w, b)
    # Expected initial velocity for weights and biases
    expected_dw_vel_init = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
    expected_db_vel_init = [np.array([[0, 0]]), np.array([[0]])]
   
    # Verify the shapes of obtained velocities
    for exp_dw, ob_dw, exp_db, obt_db in zip(expected_dw_vel_init, obtained_dw_vel_init, expected_db_vel_init, obtained_db_vel_init):
        assert(exp_dw.shape == ob_dw.shape)
        assert(exp_db.shape == obt_db.shape)

#========================================================================================================

# Test function for verifying the initialization of weights and biases for Stochastic gradient with Adam optimizer.
def test_init_adam():
    """
    
    # Purpose of test: Verify the functionality of init_adam to initialize the velocity, momentum, and squared gradient variables for the weights and biases in a neural network.

    # Input: Numpy arrays of weights and biases of the neural network.
            w = [np.array([[0.2, 0.5], [0.54, 1.6]]), np.array([[0.89], [0.45]])]
            b = [np.array([[0.7, 0.9]]), np.array([[0.2]])]
    # Expected Output: Numpy arrays list of initial velocity of weights and biases, and initial squared gradient of the weights and biases, initialized as zero arrays.
            expected_dw_vel_init = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
            expected_db_vel_init = [np.array([[0, 0]]), np.array([[0]])]
            expected_dw_s_init = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
            expected_db_s_init = [np.array([[0, 0]]), np.array([[0]])]
    
    Command to run file: pytest test_optimization_algorithms.py
    
    # Obtained Output: The actual numpy arrays list of initial velocity of weights and bias and, initial squared gradient of the weights and biases obtained from init_adam function, have the same shape as weights and bias respectively.

    """
    # Initial simple array weights and biases
    w = [np.array([[0.2, 0.5], [0.54, 1.6]]), np.array([[0.89], [0.45]])]
    b = [np.array([[0.7, 0.9]]), np.array([[0.2]])]

    # Calling the function to initialize parameters for Adam optimizer
    obtained_dw_vel_init, obtained_db_vel_init, obtained_s_dw_init, obtained_s_db_init = init_adam(w, b)
    # Expected initial velocity and square gradients for weights and biases
    expected_dw_vel_init = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
    expected_db_vel_init = [np.array([[0, 0]]), np.array([[0]])]
    expected_dw_s_init = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
    expected_db_s_init = [np.array([[0, 0]]), np.array([[0]])]
    # Verify the shapes of obtained parameters
    for exp_dw, ob_dw, exp_db, obt_db in zip(expected_dw_vel_init, obtained_dw_vel_init, expected_db_vel_init, obtained_db_vel_init):
        assert(exp_dw.shape == ob_dw.shape) # Verify shape of weight velocity
        assert(exp_db.shape == obt_db.shape)    # Verify shape of bias velocity
    for exp_dw, ob_dw, exp_db, obt_db in zip(expected_dw_s_init, obtained_s_dw_init, expected_db_s_init, obtained_s_db_init):
        assert(exp_dw.shape == ob_dw.shape) # Verify shape of weight squared gradients
        assert(exp_db.shape == obt_db.shape)    # Verify shape of bias squared gradients
#========================================================================================================

# Test function for verifying the gradient descent with momentum optimization.
def test_gradient_descent_with_momentum():
    """
    # Purpose of test: Verify the functionality of gradient_descent_with_momentum, which perform a single step of gradient descent with momentum on the weights and biases of a neural network.

    # Input: Numpy arrays lists of velocity of the weights and biases, weights and biases of the neural network, gradients of weights and biases, momentum parameter, learning rate, and time step.
            w = [np.array([[0.15, 0.5], [-0.4, 0.56]]), np.array([[0.39], [0.86]])]
            b = [np.array([[0.33, 0.21]]), np.array([[0.23]])]
            v_dw = [np.array([[0.05, 0.5], [0.2, 0.06]]), np.array([[0.9], [0.42]])]
            v_db = [np.array([[0.03, 0.1]]), np.array([[0.6]])]
            dw = [np.array([[0.21, 0.6], [0.1, 0.6]]), np.array([[0.02], [0.12]])]
            db = [np.array([[0.54, 0.7]]), np.array([[0.8]])]
            mom_para, alpha, t = 0.99, 0.23, 5   
    
    Command to run file: pytest test_optimization_algorithms.py
    
    # Expected Output: Numpy arrays list of updated velocity of weights and biases, and updated weights and biases.
            expected_v_dw = [np.array([[0.0516, 0.501 ], [0.199 , 0.0654]]), np.array([[0.8912], [0.417 ]])]
            expected_v_db = [np.array([[0.035, 0.10]]), np.array([[0.602]])]
            expected_w = [np.array([[ 0.138,  0.384 ], [-0.445 ,  0.544]]), np.array([[0.185], [0.764]])]
            expected_b = [np.array([[0.321, 0.185 ]]), np.array([[0.091]])]
    # Obtained Output: The actual numpy arrays list of updated velocity of weights and biases, and updated weights and biases obtained from gradient_descent_with_momentum function, the expected updated velocity match with the obtained ones within a tolerance of 1e-3. 

    """
    
    # Initial weights, biases, velocity, gradients, momentum parameter, learning rate, and time step
    w = [np.array([[0.15, 0.5], [-0.4, 0.56]]), np.array([[0.39], [0.86]])]
    b = [np.array([[0.33, 0.21]]), np.array([[0.23]])]
    v_dw = [np.array([[0.05, 0.5], [0.2, 0.06]]), np.array([[0.9], [0.42]])]
    v_db = [np.array([[0.03, 0.1]]), np.array([[0.6]])]
    dw = [np.array([[0.21, 0.6], [0.1, 0.6]]), np.array([[0.02], [0.12]])]
    db = [np.array([[0.54, 0.7]]), np.array([[0.8]])]
    mom_para, alpha, t = 0.99, 0.23, 5   
    
    # Calling the function to perform gradient descent with momentum
    obtained_v_dw, obtained_v_db, obtained_w, obtained_b = gradient_descent_with_momentum(v_dw, v_db, w, b, dw, db, mom_para, alpha, t)
    
    # Expected updated velocity, weights, and biases
    expected_v_dw = [np.array([[0.0516, 0.501 ], [0.199 , 0.0654]]), np.array([[0.8912], [0.417 ]])]
    expected_v_db = [np.array([[0.035, 0.10]]), np.array([[0.602]])]
    expected_w = [np.array([[ 0.138,  0.384 ], [-0.445 ,  0.544]]), np.array([[0.185], [0.764]])]
    expected_b = [np.array([[0.321, 0.185 ]]), np.array([[0.091]])]
    
    # Verify the updated velocity, weights, and biases
    for exp_vdw, obt_vdw, exp_vdb, obt_vdb in zip(expected_v_dw, obtained_v_dw, expected_v_db, obtained_v_db):
        assert(exp_vdw - obt_vdw < 1e-3).all() # Verify updated weight velocity
        assert(exp_vdb - obt_vdb < 1e-3).all()  # Verify updated bias velocity
    for exp_w, obt_w, exp_b, obt_b in zip(expected_w, obtained_w, expected_b, obtained_b):
        assert(exp_w - obt_w < 1e-3).all()  # Verify updated weights
        assert(exp_b - obt_b < 1e-3).all()   # Verify updated biases
#========================================================================================================
 # Test function for verifying the gradient descent with RMSProp optimization.
def test_gradient_descent_with_rmsprop():
    """ 
    # Purpose of test: Verify the functionality of gradient_descent_with_rmsprop, which performs a single step of gradient descent with RMSProp on the weights and biases of a neural network.

    # Input: Numpy arrays lists of squared gradient of the weights and biases, weights and biases of the neural network, gradients of weights and biases, RMSProp parameter, RMSProp epsilon value, and learning rate.
            w = [np.array([[0.25, 0.3], [0.2, 0.6]]), np.array([[0.3], [0.6]])]
            b = [np.array([[0.13, 0.4]]), np.array([[0.3]])]
            s_dw = [np.array([[0.5, 0.2], [0.2, 0.16]]), np.array([[0.3], [0.42]])]
            s_db = [np.array([[0.4, 0.5]]), np.array([[0.16]])]
            dw = [np.array([[0.12, 0.5], [0.1, 0.6]]), np.array([[0.02], [0.12]])]
            db = [np.array([[0.4, 0.3]]), np.array([[0.8]])]
            rms_para, eps, alpha = 0.9, 1e-2, 0.06
    
     Command to run file: pytest test_optimization_algorithms.py
     
    # Expected Output: Numpy arrays list of updated squared gradient of the weights and biases, and updated weights and biases.
            expected_s_dw = [np.array([[0.451, 0.205], [0.181  , 0.18]]), np.array([[0.270], [0.37944]])]
            expected_s_db = [np.array([[0.376, 0.459]]), np.array([[0.208]])]
            expected_w = [np.array([[0.239, 0.235], [0.186, 0.517]]), np.array([[0.297], [0.588]])]
            expected_b = [np.array([[0.091, 0.37]]), np.array([[0.197]])]
    # Obtained Output: The actual numpy arrays list of updated squared gradient of weights and biases, and updated weights and biases obtained from gradient_descent_with_rmsprop function, the expected updated squared gradient of the weights and biases match with the obtained ones within a tolerance of 1e-3.

    """
    # Initial weights, biases, RMSProp cache, gradients, RMSProp parameter, epsilon, and learning rate
    w = [np.array([[0.25, 0.3], [0.2, 0.6]]), np.array([[0.3], [0.6]])]
    b = [np.array([[0.13, 0.4]]), np.array([[0.3]])]
    s_dw = [np.array([[0.5, 0.2], [0.2, 0.16]]), np.array([[0.3], [0.42]])]
    s_db = [np.array([[0.4, 0.5]]), np.array([[0.16]])]
    dw = [np.array([[0.12, 0.5], [0.1, 0.6]]), np.array([[0.02], [0.12]])]
    db = [np.array([[0.4, 0.3]]), np.array([[0.8]])]
    rms_para, eps, alpha = 0.9, 1e-2, 0.06
    
    # Calling the function to perform gradient descent with RMSProp
    obtained_s_dw, obtained_s_db, obtained_w, obtained_b = gradient_descent_with_rmsprop(s_dw, s_db, w, b, dw, db, rms_para, eps, alpha)
    
    # Expected updated RMSProp weights, and biases
    expected_s_dw = [np.array([[0.451, 0.205], [0.181  , 0.18]]), np.array([[0.270], [0.37944]])]
    expected_s_db = [np.array([[0.376, 0.459]]), np.array([[0.208]])]
    expected_w = [np.array([[0.239, 0.235], [0.186, 0.517]]), np.array([[0.297], [0.588]])]
    expected_b = [np.array([[0.091, 0.37]]), np.array([[0.197]])]
    
    # Verify updated RMSProp weights, and biases
    for exp_vdw, obt_vdw, exp_vdb, obt_vdb in zip(expected_s_dw, obtained_s_dw, expected_s_db, obtained_s_db):
        assert(exp_vdw - obt_vdw < 1e-3).all()  # Verify updated updated squared gradient of the weights
        assert(exp_vdb - obt_vdb < 1e-3).all()  # Verify updated updated squared gradient of the biases
    for exp_w, obt_w, exp_b, obt_b in zip(expected_w, obtained_w, expected_b, obtained_b):
        assert(exp_w - obt_w < 1e-3).all()   # Verify updated weights
        assert(exp_b - obt_b < 1e-3).all()  #  # Verify updated biases
#========================================================================================================

# Test function for verifying the gradient descent with Adam optimization.

def test_gradient_descent_with_adam():
    """
    
    # Purpose of test: Verify the functionality of gradient_descent_with_adam, which performs a single step of gradient descent with Adam on the weights and biases of a neural network.

    # Input: Numpy arrays lists of velocity of the weights and biases, squared gradient of the weights and biases, weights and biases of the neural network, gradients of weights and biases, RMSProp parameter, RMSProp epsilon value, time step and learning rate.
            w = [np.array([[0.15, 0.5], [-0.4, 0.56]]), np.array([[0.39], [0.86]])]
            b = [np.array([[0.33, 0.21]]), np.array([[0.23]])]
            v_dw = [np.array([[0.05, 0.5], [0.2, 0.06]]), np.array([[0.9], [0.42]])]
            v_db = [np.array([[0.03, 0.1]]), np.array([[0.6]])]
            s_dw = [np.array([[0.5, 0.2], [0.2, 0.16]]), np.array([[0.3], [0.42]])]
            s_db = [np.array([[0.4, 0.5]]), np.array([[0.16]])]
            dw = [np.array([[0.21, 0.6], [0.1, 0.6]]), np.array([[0.02], [0.12]])]
            db = [np.array([[0.54, 0.7]]), np.array([[0.8]])]
            mom_para, alpha, t, rms_para, eps = 0.99, 0.0023, 5, 0.9, 1e-2
    
    Command to run file: pytest test_optimization_algorithms.py
    
    # Expected Output: Numpy arrays list of updated weights and biases.
            expected_w = [np.array([[ 0.14, 0.467], [-0.413,  0.555]]), np.array([[0.338],[0.839]])]
            expected_b = [np.array([[0.328, 0.205]]), np.array([[0.190]])]
    # Obtained Output: The actual numpy arrays list of updated weights and biases obtained from gradient_descent_with_adam function, the expected updated weights and biases match with the obtained ones within a tolerance of 1e-3.
 
    """
    # Initial weights, biases, velocities, squared gradient, gradients, momentum parameter, epsilon, and time step
    w = [np.array([[0.15, 0.5], [-0.4, 0.56]]), np.array([[0.39], [0.86]])]
    b = [np.array([[0.33, 0.21]]), np.array([[0.23]])]
    v_dw = [np.array([[0.05, 0.5], [0.2, 0.06]]), np.array([[0.9], [0.42]])]
    v_db = [np.array([[0.03, 0.1]]), np.array([[0.6]])]
    s_dw = [np.array([[0.5, 0.2], [0.2, 0.16]]), np.array([[0.3], [0.42]])]
    s_db = [np.array([[0.4, 0.5]]), np.array([[0.16]])]
    dw = [np.array([[0.21, 0.6], [0.1, 0.6]]), np.array([[0.02], [0.12]])]
    db = [np.array([[0.54, 0.7]]), np.array([[0.8]])]
    mom_para, alpha, t, rms_para, eps = 0.99, 0.0023, 5, 0.9, 1e-2
    # Calling the function to perform gradient descent with Adam
    obtained_w, obtained_b = gradient_descent_with_adam(v_dw, v_db, s_dw, s_db, w, b, dw, db, mom_para, rms_para, alpha, eps, t)
    
    # Expected updated weights and biases
    expected_w = [np.array([[ 0.14, 0.467], [-0.413,  0.555]]), np.array([[0.338],[0.839]])]
    expected_b = [np.array([[0.328, 0.205]]), np.array([[0.190]])]
    
    # Verify the updated weights and biases
    for exp_w, obt_w, exp_b, obt_b in zip(expected_w, obtained_w, expected_b, obtained_b):
        assert(exp_w - obt_w < 1e-3).all()
        assert(exp_b - obt_b < 1e-3).all()
#========================================================================================================

# Test function for verifying the mini-batches creation.
def test_mini_batches():
    """
    # Purpose of test: Verify the functionality of mini_batches, which splits the training data into mini-batches.

    # Input: Numpy arrays independent training and test datasets, and mini-batch size.
            x_train = np.random.rand(110, 5)
            y_train = np.random.rand(110, 1)
            mini_batch_size = 20 
    # Expected Output: A list of numpy arrays of mini-batches with shape (mini_batch_size, num_features) but the shape of last mini-batch is (n_samples - (mini_batch_size * n_mini_batches), num_features).
        
    Command to run file: pytest test_optimization_algorithms.py
    
    # Obtained Output: The actual numpy arrays list of mini-batches obtained from mini_batches function, where the length of x_train_mini_batch and y_train_mini_batch are equal, 
    #                   the last mini batch should have a length of x_train.shape[0] % mini_batch_size, all the other mini-batched have a lenth of given mini-batch size.
        
    """
    # Create random data for independent training features and training labels
    x_train = np.random.rand(110, 5)
    y_train = np.random.rand(110, 1)
    mini_batch_size = 20    # Mini-batch size
    
    # Calling the function to create mini-batches
    x_train_mini_batch, y_train_mini_batch = mini_batches(x_train, y_train, mini_batch_size)
    
    # Verifying that the number of mini-batches is consistent between independent training features and training labels
    assert(len(x_train_mini_batch) == len(y_train_mini_batch))
    # Verifying that the last mini-batch contains the leftover samples
    assert(len(x_train_mini_batch[-1]) == len(y_train_mini_batch[-1]) == x_train.shape[0] % mini_batch_size)
    # Verifying that all mini-batches except the last one have the given mini-batch size
    for mini_batch_x, mini_batch_y in zip(x_train_mini_batch[:-1], y_train_mini_batch[:-1]):
        assert(len(mini_batch_x) == mini_batch_size) 
        assert(len(mini_batch_y) == mini_batch_size)
    
    
#========================================================================================================

    
    
    



