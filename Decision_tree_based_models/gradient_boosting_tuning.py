"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 10 May 2024
Description: Tuning the hyperparameters like number of trees, maximum depth of tree and minimum samples required to split.

"""

# Import necessary library and functions
import numpy as np
import os
import sys

ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ANN'))
sys.path.insert(0, ANN_dir)
from gradient_boosting_regression import gradient_boosting_main
from ANN_regression import read_dataset_reg, split_dataset


def gbparameters(x_train, x_test, y_train, y_test):
    """
    Perform hyperparameter tuning of number of trees, maximum depth of tree and minimum samples to split for gradient boosting model.

    Args:
    file (str): Path to the CSV file containing the dataset.

    Returns:
    None
    """
    r2_error = 0
    hyper_par_list = []
    optimum_values = ()
    # Iterate over different ranges of hyperparameters
    for trees_n in np.linspace(10, 20, 2):
        
            for samples_min in np.linspace(10, 20, 2):
                
                for depth_max in range(1, 5):
                    print(trees_n, samples_min, depth_max)
                    
                    # Run the main function with different hyperparameters
                    _,_,_,opt_r2sqr = gradient_boosting_main(x_train, x_test, y_train, y_test, trees_n, depth_max, samples_min )
                    #print(opt_f1_score)
                    hyper_par_list.append([trees_n, samples_min, depth_max, opt_r2sqr])
                    # Update optimal hyperparameters if F1 score is higher
                    if opt_r2sqr > r2_error:
                        r2_error = opt_r2sqr
                        optimum_values = (trees_n, samples_min, depth_max, opt_r2sqr)
                        print(optimum_values)
    hyper_par_list = np.array(hyper_par_list)
    np.savetxt('gradient_boosting_hyperparameter.csv', hyper_par_list, delimiter=',', header='trees_n,samples_min,depth_max Batch,opt_f1_score')
    
    return None
    
file_path = os.path.join(ANN_dir, 'input_file_regression.csv')
independent_features, dependent_feature = read_dataset_reg(file_path)
x_train, x_test, y_train, y_test,_ = split_dataset(independent_features, dependent_feature, train_ratio = 0.8)
gbparameters(x_train, x_test, y_train, y_test)

