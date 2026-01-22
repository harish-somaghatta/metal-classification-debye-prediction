
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 24 Jan 2024
Description: Perform Principal Component Analysis on independent features.

"""

# Import the necessary libraries
import numpy as np
import os
import sys

def read_features(file):
    """
    Read the input data from a CSV file.

    Arguments:
    file (str): Path of the file containing the input data.

    Returns:
    all_features (numpy.ndarray): All features from the CSV file.
    
    """
    # Load all the features from the csv file
    all_features = np.genfromtxt(file, delimiter=",", skip_header=1)
    
    return all_features

def feature_scaling(all_features):
    """
    Perform feature scaling using Z-score method on the independent features.

    Arguments:
    all_features (numpy.ndarray): All features including dependent features.

    Returns:
    independent_features (numpy.ndarray): Independent features (Excluding last two columns).
    features_scaled (numpy.ndarray): Scaled features using Z-score scaling.
    
    """
    no_features = all_features.shape[1]  # Get the number of features (columns).
    excluded_cols = [no_features - 2, no_features - 1] # Exclude dependent(target) features.
    # Get the independent features by removing the last two target features.
    independent_features = np.delete(all_features, excluded_cols, axis=1)
    # Calculate the mean for each independent features.
    mean_col = np.mean(independent_features, axis=0)
    # Calculate the standard deviation for each independent features.
    std_dev_col = np.std(independent_features, axis=0)
    std_dev_col[std_dev_col == 0] = 1 # To excluded 0/0 error.
    # Scale independent features using Z-score scaling method.
    features_scaled = (independent_features - mean_col) / std_dev_col
    
    # Return independent features and scaled features.
    return independent_features, features_scaled


def sorted_eigval_eigvec(scaled_fea):
    """
    Compute eigen values and eigen vectors, and sort them in descending order.
    
    Arguments:
    scaled_fea (numpy.ndarray): Scaled features obtained from feature scaling.
    
    Returns:
    sort_eig_val (numpy.ndarray): Sorted eigenvalues in descending order.
    sort_eig_vec (numpy.ndarray): Sorted eigenvectors corresponding to the eigenvalues.
    
    """
    
    # Compute the covariance matrix(each column - variable)
    mat_cov = np.cov(scaled_fea, rowvar=False)
    
    # Get the eigen values and eigen vectors
    eig_val, eig_vec = np.linalg.eigh(mat_cov)
    # Sorting the eigen values and the corresponding eigen vectors in descending order.
    sort_indices = np.argsort(-eig_val)
    sort_eig_val = eig_val[sort_indices]
    sort_eig_vec = eig_vec[:, sort_indices]
    
    # Returns sorted eigen values and the corresponding eigen vectors.
    return sort_eig_val, sort_eig_vec

def keisers_rule(sort_eig_val):
    """
    Calculate the number of pricipal components based on Keiser's rule(Variance > 1)

    Arguments:
    sort_eig_val (numpy.ndarray): Sorted eigen values in descending order
    
    Return:
    num_pc (int): Number of pricipal components based on Keiser's rule

    """
    # The eigen values with variance > 1
    keiser_pc = sort_eig_val > 1 
    #print(keiser_pc)
    # The number of eigen values with variance > 1
    num_pc = np.sum(keiser_pc)
    
    return num_pc

def variance_explained(scaled_fea, eigen_val, threshold):
    """
    Calculate the number of principal components required to explain variance above a certain threshold.

    Parameters:
    scaled_fea (numpy.ndarray): Scaled feature.
    eigen_val (numpy.ndarray): Eigenvalues of the covariance matrix.
    threshold (float): Variance explained threshold.

    Returns:
    num_pc (int): Number of principal components required to explain variance above the threshold.
    """
    # Calculate the variance of each feature
    variance = np.var(scaled_fea, axis = 0)
    total_variance = np.sum(variance)
    #print(total_variance)
    # Calculate the variance explained by each eigenvalue
    variance_explained = eigen_val / total_variance
    #print("variance_explained: ", variance_explained)
    # Calculate the cumulative variance explained
    variance_exp_cum = np.cumsum(variance_explained)
    #print(variance_exp_cum)
    # Calculate the number of principal components required to explain variance above the threshold
    var_exp_pc = variance_explained > threshold
    num_pc = np.sum(var_exp_pc)
    
    return num_pc
    
    
def pca(scaled_ind_features, sort_eig_vectors, num_pc):
    """
    Perform Principal Component Analysis on the independent features.
    
    Arguments:
    all_features (np.ndarray): All features including dependent features.
    scaled_ind_features (np.ndarray): Scaled independent features.
    sort_eig_vectors(np.ndarray): Sorted eigen vectors corresponding to the eigen values in descending order.
    num_pc(int): Number of principal components according to Kaiser's rule.
    
    Returns:
    principal_components(np.ndarray): Returns principal components after PCA.
    """
    
    # Extract the eigen vectors based on the number of components chosen.
    extracted_eigen_vec = sort_eig_vectors[:, :num_pc]
    
    # Transform the original features (Independent features) to principal components.
    principal_components = np.dot(scaled_ind_features, extracted_eigen_vec)
    
    # Return principal components
    return principal_components

def write_pc_to_csv(all_features, num_pc, principal_comp, output):
    """
    Writes principal components and dependent features to a CSV file.
    
    Arguments:
    all_features (np.ndarray): All features including dependent features.
    num_pc(int): Number of principal components according to Keiser's rule.
    principal_comp (np.ndarray): Principal components after PCA.
    
    Returns:
    None 
    
    """
    
    # Append the dependent features of all_features
    dependent_features = all_features[:, -2:]
    principal_comp_dep_fea = np.hstack((principal_comp, dependent_features))
    
    # Header of dependent features
    dependent_fea_header = ["material_type", "debye_temperature"]
    # header names based on the number of principal components.
    header_list = [f"pca_{i+1}" for i in range(num_pc)] + dependent_fea_header
    header = np.array(header_list, dtype = str) # List to ndarray
    # Stack the header over the principal components vertically.
    pca_data_dependent_fea = np.vstack((header, principal_comp_dep_fea))
    
    # Write the resulted principal components and dependent features to the csv file.
    np.savetxt(output, pca_data_dependent_fea, delimiter=',', fmt = "%s")
    

# Drop the columns in the csv file which contains only zeros.
def drop_null_val_col(file):
    """
    Drop the columns in the csv file which contains only zeros.

    Arguments:
    file (str): Path of the input CSV file.

    Returns:
    None (Writes modified data to a new file 'pca_data.csv').
    
    """  
    # Read the file
    lines = np.genfromtxt(file, delimiter=",", dtype=str)
    header = lines[0]
    data = lines[1:]

    # Check for non-zero columns along columns(axis 0).
    non_zero_columns = np.any(data != "0", axis=0)
    # Get the non-zero indices where non-zero columns are true.
    non_zero_col_ind = np.where(non_zero_columns)[0]

    with open("non_zero_columns.csv", "w") as output_file:
        # Write only non-zero header names to the file.
        modified_header = header[non_zero_col_ind]
        output_file.write(",".join(modified_header) + "\n")

        for each_line in data:
            modified_line = each_line[non_zero_col_ind]
            output_file.write(",".join(modified_line) + "\n")

def main_pca(file_path, op_file):

    # Drop the columns that contain only zeros.
    drop_null_val_col(file_path)
    
    #  Load all the features from the input file.
    all_features = read_features("non_zero_columns.csv")
    
    # Get the scaled features and independent features.
    independent_features, scaled_features = feature_scaling(all_features)
    
    # Get the sorted eigen values and the corresponding eigen vectors.
    sorted_eigval, sorted_eigvec = sorted_eigval_eigvec(scaled_features)
    
    # Get the number of princial components using Kaiser's rule.
    no_principal_components = keisers_rule(sorted_eigval)
    #print(no_principal_components)
    
    # Perform Principal Component Analysis and returns principal components.
    principal_components = pca(scaled_features, sorted_eigvec, no_principal_components)
    
    # Writes principal components and dependent features to the csv file.
    write_pc_to_csv(all_features, no_principal_components, principal_components, op_file)
    
if __name__ == "__main__":
    
    # Get the relative path of the csv file
    data_preprocessing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preprocessing'))
    sys.path.insert(0, data_preprocessing_dir)
    input_file = os.path.join(data_preprocessing_dir, 'DPP5_label_ordinal_encode_file.csv')
    output_file = "principal_components_with_dependent_features.csv"
    main_pca(input_file, output_file)


