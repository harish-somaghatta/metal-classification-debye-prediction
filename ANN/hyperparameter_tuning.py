"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 04 Apr 2024
Description: Tuning the hyperparameters like learning rate, number of neurons in each hidden layer, mini-batch size.

"""
import numpy as np
from ANN_classification import ANN_classification_main
from ANN_regression import main_reg

def initial_alpha_tuning(val):
    """
    Perform hyperparameter tuning of learning rate.

    Parameters:
    val (int): Number of different learning rate values.

    Returns:
    cost_alpha_final (float): Optimal learning rate value corresponding to the minimum cost.
    opt_acc_alpha (float): Optimal learning rate value corresponding to the maximum training accuracy.
    opt_test_acc_alpha (float): Optimal learning rate value corresponding to the maximum test accuracy.
    opt_train_pre_alpha (float): Optimal learning rate value corresponding to the maximum training precision.
    opt_test_pre_alpha (float): Optimal learning rate value corresponding to the maximum test precision.
    opt_train_recall_alpha (float): Optimal learning rate value corresponding to the maximum training recall.
    opt_test_rec_alpha (float): Optimal learning rate value corresponding to the maximum test recall.
    opt_train_f1_alpha (float): Optimal learning rate value corresponding to the maximum training F1 score.
    opt_test_f1_alpha (float): Optimal learning rate value corresponding to the maximum test F1 score.
    
    """
    
    # Initialize lists to store cost and evaluation metrics values.
    alpha_values = []
    cost_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    train_precision_list = []
    test_precision_list = []
    train_recall_list = []
    test_recall_list = []
    train_f1_list = []
    test_f1_list = []
    
    hyperparameters_list = []
    
    # Get the random values for learning rate
    #power_val = np.random.rand(val) * -2   #  For sgd and sgdm
    power_val = np.random.rand(val) * -4    # For RMSProp and Adam
    alpha_val = 10 ** power_val
    n_neurons = [30, 50]
    # Iterate over each learning rate value
    for i, single_val in enumerate(alpha_val):
        #print("Parameter No. ", i)
        #print("alpha val", single_val)
        cost, train_acc, test_acc, train_pre, test_pre, tr_rec, te_rec, tr_f, te_f = ANN_classification_main(15000, single_val, 256, n_neurons)
        #print("Cost value is: ",cost)
        alpha_values.append(single_val)
        # Consider high cost value if the cost is not numeric.
        if np.isnan(cost):
            cost = 100
        
        # Append cost and evaluation metrics to the corresponding list.
        cost_list.append(cost)
        train_accuracy_list.append(train_acc)
        test_accuracy_list.append(test_acc)
        train_precision_list.append(train_pre)
        test_precision_list.append(test_pre)
        train_recall_list.append(tr_rec)
        test_recall_list.append(te_rec)
        train_f1_list.append(tr_f)
        test_f1_list.append(te_f)
    
    # Get the index of the best cost and evaluation metric values.
    opt_cost = np.argmin(cost_list)
    opt_train_accuracy = np.argmax(train_accuracy_list)
    opt_test_accuracy = np.argmax(test_accuracy_list)
    opt_train_precision = np.argmax(train_precision_list)
    opt_test_precision = np.argmax(test_precision_list)
    opt_train_recall = np.argmax(train_recall_list)
    opt_test_recall = np.argmax(test_recall_list)
    opt_train_f1 = np.argmax(train_f1_list)
    opt_test_f1 = np.argmax(test_f1_list)
    
    # Get the optimal learning rate value for cost and evalution metric.
    cost_alpha_final = alpha_values[opt_cost]
    opt_acc_alpha = alpha_values[opt_train_accuracy]
    opt_test_acc_alpha = alpha_values[opt_test_accuracy]
    opt_train_pre_alpha = alpha_values[opt_train_precision]
    opt_test_pre_alpha = alpha_values[opt_test_precision]
    opt_train_recall_alpha = alpha_values[opt_train_recall]
    opt_test_rec_alpha = alpha_values[opt_test_recall]
    opt_train_f1_alpha = alpha_values[opt_train_f1]
    opt_test_f1_alpha = alpha_values[opt_test_f1]
    
    print(alpha_values, cost_list, train_accuracy_list, test_accuracy_list, train_precision_list, test_precision_list, train_recall_list, test_recall_list, train_f1_list, test_f1_list)
    hyperparameters_list.append([alpha_values, cost_list, train_accuracy_list, test_accuracy_list, train_precision_list, test_precision_list, train_recall_list, test_recall_list, train_f1_list, test_f1_list])
    # Return optimal learning rate value of the corresponding cost and metric value.
    return cost_alpha_final, opt_acc_alpha,opt_test_acc_alpha,opt_train_pre_alpha,opt_test_pre_alpha, opt_train_recall_alpha, opt_test_rec_alpha, opt_train_f1_alpha, opt_test_f1_alpha

#alpha_values = initial_alpha_tuning(20)

#alpha_values = np.array(alpha_values)
# Save the hyperparemeters to csv file
#np.savetxt('hyperparameter_tuning_alpha_val_sgd.csv', alpha_values, delimiter=',', header='alpha_values, cost_list, train_accuracy_list, test_accuracy_list, train_precision_list, test_precision_list, train_recall_list, test_recall_list, train_f1_list, test_f1_list')
#np.savetxt('hyperparameter_tuning_alpha_val_sgdm.csv', alpha_values, delimiter=',', header='alpha_values, cost_list, train_accuracy_list, test_accuracy_list, train_precision_list, test_precision_list, train_recall_list, test_recall_list, train_f1_list, test_f1_list')
#np.savetxt('hyperparameter_tuning_alpha_val_rmsprop.csv', alpha_values, delimiter=',', header='alpha_values, cost_list, train_accuracy_list, test_accuracy_list, train_precision_list, test_precision_list, train_recall_list, test_recall_list, train_f1_list, test_f1_list')
#np.savetxt('hyperparameter_tuning_alpha_val_adam.csv', alpha_values, delimiter=',', header='alpha_values, cost_list, train_accuracy_list, test_accuracy_list, train_precision_list, test_precision_list, train_recall_list, test_recall_list, train_f1_list, test_f1_list')


def opt_num_layers_neurons_batch_size_(model):
    """
    Get the optimimum number of layers, neurons, and batch size.

    Returns:
    opt_cost (float): Optimal cost value obtained.
    opt_neurons (list): Optimal number of neurons for the hidden layers.
    
    """
    # Initialize optimal cost and number of neurons list
    opt_cost = 100
    opt_neurons = []
    # Different ranges for number of neurons and mini-batch size
    n_neurons_range = [4, 15, 30, 50]
    mini_batch_range = [256, 512, 1024, 2048]
    opt_values = []
    
    # Iterate over different combinations of neurons and mini-batch sizes
    for n_neurons1 in n_neurons_range:
        for n_neurons2 in n_neurons_range:
            for m_batch in mini_batch_range:
                current_neurons = [n_neurons1, n_neurons2]
                print("N1, N2, minibatch", [n_neurons1, n_neurons2], m_batch)
                
                # Run the main function with different combinations
                if model == "classification":
                    cost, _, _, _, _, _, _, _, _ = main(15000, m_batch, current_neurons)
                    print("cost value is: ", cost)
                    opt_values.append([n_neurons1, n_neurons2, m_batch, cost])
                    
                    # Update the cost and neurons if it has less cost value
                    if cost < opt_cost:
                        opt_cost = cost
                        opt_neurons = current_neurons
                if model == "regression":
                    cost = main_reg(15000, 0.00354728214309162, m_batch, current_neurons)
                    print("cost value is: ", cost)
                    opt_values.append([n_neurons1, n_neurons2, m_batch, cost])
                    
                    # Update the cost and neurons if it has less cost value
                    if cost < opt_cost:
                        opt_cost = cost
                        opt_neurons = current_neurons
                    
    opt_values_arr = np.array(opt_values)
    
    return opt_values_arr

#parameter_values_class = opt_num_layers_neurons_batch_size_(model = "regression")
#np.savetxt('optimum_neuronslist.csv', parameter_values_class, delimiter=',', header='Neuron1,Neuron2,Mini Batch,Cost')
#parameter_values_reg = opt_num_layers_neurons_batch_size_(model = "regression")
#np.savetxt('optimum_neurons_regression_list.csv', parameter_values_reg, delimiter=',', header='Neuron1,Neuron2,Mini Batch,Cost')

#print(optimum_cost, opt_neurons)