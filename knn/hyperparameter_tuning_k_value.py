"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 28 Apr 2024
Description: Tune the K value of KNN classification algorithm.

"""
# Import main function from knn_classification file
from knn_classification import main_knn
import numpy as np

def tune_k_neighbours(file):
    """
    Tune the number of neighbors (k) for the k-nearest neighbors classification algorithm.

    Returns:
    k_values (list): K values and their corresponding F1 scores in a list.
    """
    
    # Initialize an empty list to store k values and corresponding f1 scores
    k_values = []
    
    # Iterate over range of k values
    for k in range(2, 21):
        # Get the f_score for the current k value
        _, f1_score = main_knn(file, k)
        #print(f1_score)
        k_values.append([k, f1_score])
        
    return k_values

input_file = "principal_components_with_dependent_features.csv"
k_neighbours = tune_k_neighbours(input_file)
# Covert list to numpy for compaitability to save 
K_values = np.array(k_neighbours)
# Save the hyperparemeters to csv file
np.savetxt('hyperparameter_tuning_k_val.csv', K_values, delimiter=',', header='k,f1_score')

    
    
    
    
