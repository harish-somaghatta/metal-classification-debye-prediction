"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 29 Apr 2024
Description: Plot a graph for the hyperparameter tuning values.

"""

import numpy as np
import matplotlib.pyplot as plt

def plotting_k_vs_f1_score(datapoints):
    """
    Plot the Hyperparameters tuning of k-values.

    Arguments:
    datapoints (numpy.ndarray): An array containing K values and corresponding F1 scores.

    Returns:
    None
    """
    # Get K values and F1 score values from the datapoints
    k_values = data[:, 0]
    f1_score = data[:, 1]
    
    # Scatter the k-values and f_score values
    plt.scatter(k_values, f1_score)
    
    # Get the index of hightest f1 score value
    higest_f1_score_idx = np.argmax(f1_score)
    # Mark the optimal k-value
    plt.scatter(k_values[higest_f1_score_idx], f1_score[higest_f1_score_idx], label='Optimal k-value')
    # Plot the connecting lines
    plt.plot(k_values, f1_score, linestyle='--', color='green')
    
    # Give labels
    plt.xlabel('K Values')
    plt.ylabel('F1 Score')
    plt.title('K-Values vs F1 Score')
    # Give legend
    plt.legend()
    # Set x-axis values as k values
    plt.xticks(k_values)
    # Show the plot
    plt.show()
    
    return None

data = np.genfromtxt('hyperparameter_tuning_k_val.csv', delimiter = ',', skip_header = True)

plotting_k_vs_f1_score(data)

