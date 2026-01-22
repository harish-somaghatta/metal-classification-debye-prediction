"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 04 May 2024
Description: Perform unit test for each function of Random Forest classification.

"""
import numpy as np
import os
import sys
from random_forest_classification import main_random_forest, split_dataset, read_dataset
   
def random_hyperparameters(x_train, x_test, y_train, y_test):
    """
    Perform hyperparameter tuning of number of trees, maximum depth of tree and minimum samples to split for  random forest model.

    Args:
    file (str): Path to the CSV file containing the dataset.

    Returns:
    None
    """
    f1_score = 0
    hyper_par_list = []
    
    # Iterate over different ranges of hyperparameters
    for trees_n in range(2, 8):
            for samples_min in np.linspace(10, 40, 4):
                
                for depth_max in np.linspace(0, 25, 6):
                    print(trees_n, samples_min, depth_max)
                    
                    # Run the main function with different hyperparameters
                    _, opt_f1_score = main_random_forest(x_train, x_test, y_train, y_test, trees_n, depth_max, samples_min, seed = 10)
                    #print(opt_f1_score)
                    hyper_par_list.append([trees_n, samples_min, depth_max, opt_f1_score])
                    # Update optimal hyperparameters if F1 score is higher
                    if f1_score > opt_f1_score:
                        f1_score = opt_f1_score
    hyper_par_list = np.array(hyper_par_list)
    np.savetxt('random_forest_hyperparameters.csv', hyper_par_list, delimiter=',', header='trees_n,samples_min,depth_max Batch,opt_f1_score')
    
    return None

pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))
sys.path.insert(0, pca_dir)
file_path = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
independent_features, target_feature = read_dataset(file_path)
x_train, x_test, y_train, y_test,_ = split_dataset(independent_features, target_feature, train_ratio = 0.8 )
random_hyperparameters(x_train, x_test, y_train, y_test)