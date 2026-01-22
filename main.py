# Main file
import numpy as np
import os
import sys

data_preprocessing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_preprocessing'))
sys.path.insert(0, data_preprocessing_dir)
pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pca'))
sys.path.insert(0, pca_dir)
ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ANN'))
sys.path.insert(0, ANN_dir)
KNN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'KNN'))
sys.path.insert(0, KNN_dir)
Dt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Decision_tree_based_models'))
sys.path.insert(0, Dt_dir)
Mlr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Multiple_linear_regression'))
sys.path.insert(0, Mlr_dir)

raw_file_path = os.path.join(data_preprocessing_dir, 'raw_compounds_data.csv')
ele_count_file_path = os.path.join(data_preprocessing_dir, 'compounds_to_ele_count.csv')
raw_elements_data_path = os.path.join(data_preprocessing_dir, 'Elements_data.csv')
updated_elements_data = os.path.join(data_preprocessing_dir, 'Elements_data_updated.csv')
DPP1_weighted_atomic_properties_file =  os.path.join(data_preprocessing_dir, 'DPP1_weighted_atomic_prop.csv')
DPP2_ind_col_file_path = os.path.join(data_preprocessing_dir, 'DPP2_ind_ele_col.csv')
DPP3_label_encode_path =  os.path.join(data_preprocessing_dir, 'DPP3_label_encode.csv')
DPP4_ele_composition_path =  os.path.join(data_preprocessing_dir, 'DPP4_ele_composition_num_ele.csv')
DPP5_ordinal_encode_path = os.path.join(data_preprocessing_dir, 'DPP5_label_ordinal_encode_file.csv')
pca_file_path = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
princiapal_components = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
only_metals_data = os.path.join(ANN_dir, 'input_file_regression.csv')


from compounds_to_element_count import main_compounds_to_ele_count
from elements_data_updated import main_ele_data_updated
from Desc2_weighted_atomic_properties import main_weighted_prop
from Desc1_comp_to_ind_ele import main_des_2
from label_encoding import main_label_encode
from one_hot_encode_ele_composition import main_encode_ele_comp
from ordinal_encod import main_ordinal_label_enode
from pca import main_pca
from ANN_classification import read_dataset, split_dataset, ANN_classification_main
from knn_classification import main_knn
from random_forest_classification import main_random_forest
from ANN_regression import read_dataset_reg, main_reg
from gradient_boosting_regression import gradient_boosting_main
from multiple_linear_regression import mlr_main

def main_fn():
    
    # Get elements and their count from chemical formulas of compounds(Raw compounds file).
    print("Step 1: Extracting elements and their counts from chemical formulas.")
    main_compounds_to_ele_count(raw_file_path, ele_count_file_path)
    
    # Fill the missing values of atomic properties with their respective mean values
    print("\nStep 2: Filling missing values of atomic properties.")
    main_ele_data_updated(raw_elements_data_path, updated_elements_data)
    
    # Create Descriptor 1 (Converts atomic properties to weighted atomic properties)
    print("\nStep 3: Creating Descriptor 1.")
    main_weighted_prop(raw_file_path, ele_count_file_path, updated_elements_data, DPP1_weighted_atomic_properties_file)
    
    # Create Descriptor 2
    print("\nStep 4: Creating Descriptor 2.")
    main_des_2( ele_count_file_path , updated_elements_data, DPP1_weighted_atomic_properties_file, DPP2_ind_col_file_path)
    
    # label encoding of material type(Metal - 1, Non-Metal - 0)
    print("\nStep 5: Label encoding of material type.")
    main_label_encode(DPP2_ind_col_file_path, DPP3_label_encode_path)
    
    # One hot encode and element composition vector
    print("\nStep 6: One-hot encoding of number of compounds in the compound formula")
    main_encode_ele_comp(ele_count_file_path, DPP3_label_encode_path, DPP4_ele_composition_path)
    
    # Ordinal encoding of thermal conductivity
    print("\nStep 7: Ordinal encoding of thermal conductivity.")
    main_ordinal_label_enode(DPP4_ele_composition_path, DPP5_ordinal_encode_path)
    
    # Perform Principal Component Analysis
    print("\nStep 8: Performing Principal Component Analysis (PCA).")
    main_pca(DPP5_ordinal_encode_path, pca_file_path)
    
    print("\nStep 9: Reading the principal components and splitting into independent and dependent features.")
    independent_features, dependent_features = read_dataset(princiapal_components)
    
    print("\nStep 10: Splitting the dataset into training and testing datasets.")
    x_train, x_test, y_train, y_test, targetf_regression = split_dataset(independent_features, dependent_features, train_ratio = 0.8 )
    
    print("\nInformation about the dataset:")
    print(f"Number of features: {independent_features.shape[1]}")
    print(f"Number of samples: {independent_features.shape[0]}")
    print(f"Training set size: {x_train.shape[0]}")
    print(f"Test set size: {x_test.shape[0]}")
    
    
    print("\nPlease choose the classification algorithms:")
    print("1 for Deep Neural Network")
    print("2 for K-Nearest Neighbours")
    print("3 for Random Forest Classification")
    classification_choice = input("\nEnter your choice 1/2/3: ").strip()
    
    if classification_choice == '1':
        epoch, m_batch, alpha = 15000, 256, 0.000354728214309162
        n_neurons = [30, 50]
        ANN_classification_main(x_train, x_test, y_train, y_test,epoch, alpha, m_batch, n_neurons, targetf_regression)
        
    elif classification_choice == '2':
        k = 3
        prediction, _ = main_knn(x_train, x_test, y_train, y_test, k)
        
    elif classification_choice == '3':
        num_trees, maximum_depth, minimum_samples, seed = 7, 20, 0, 10
        main_random_forest(x_train, x_test, y_train, y_test, num_trees, maximum_depth, minimum_samples, seed)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\n Please enter 0 if you want to quit or enter 1 if you want to predict Debye temperature using regression models")
    
    
    reg_model = input("0 or 1: ").strip()
    
    if reg_model == '0':
        print("The material data is classified into metals and non-metals. \nThe metals data is split into training and test datset for regression task")
    elif reg_model == '1':
    
        
        independent_features_reg, dependent_feature_reg = read_dataset_reg(only_metals_data)
        x_train, x_test, y_train, y_test,_ = split_dataset(independent_features_reg, dependent_feature_reg, train_ratio = 0.8)
        
        print("\nPlease choose the regression algorithms:")
        print("1 for Gradient Boosting")
        print("2 for Artificial Neural Network")
        print("3 for Multiple Linear Regression")
        regression_choice = input("\nEnter your choice 1/2/3: ").strip()
        
        if regression_choice == '1':
            num_trees, max_depth, min_samples = 10, 20, 4
            gradient_boosting_main(x_train, x_test, y_train, y_test, num_trees, max_depth, min_samples)
        
        elif regression_choice == '2':
            epoch, m_batch, alpha = 15000, 256, 0.00354728214309162
            n_neurons = [4]
            main_reg(x_train, x_test, y_train, y_test, epoch, alpha, m_batch, n_neurons)
        
        elif regression_choice == '3':
            iterations = 10000
            mlr_main(x_train, x_test, y_train, y_test, iterations)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    else:
        print("Invalid choice. Please enter 0 or 1.")
            
            
            
    
if __name__ == "__main__":
    # Detailed instructions for the user.
    print("\nWelcome to the Material Classification and Debye temperature prediction Program!")
    print("This program will guide you through the process.")
    print("Please follow the prompts to proceed.\n")
    
    # Execute the main function.
    main_fn()