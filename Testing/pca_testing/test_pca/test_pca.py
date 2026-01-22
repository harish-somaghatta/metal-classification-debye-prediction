"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 28 Jan 2024
Description: Perform unit test for the functions used in Principal Component Analysis.
"""
import os
import sys
import numpy as np
pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..','pca'))
sys.path.insert(0, pca_dir)
from pca import read_features, feature_scaling, sorted_eigval_eigvec, keisers_rule, variance_explained, pca

#===============================================================================================================

def test_overall_variance():
    """
    # Purpose of test: Verify whether the total variance of the input data is equal to the sum of all eigenvalues obtained from the principal component analysis (PCA).

    # Input: The input file "non_zero_columns.csv" contains all the non-zero features.
    
    Command to run file: pytest test_pca.py
    
    # Expected output: The total variance of the input data  matches the sum of all eigenvalues obtained from PCA which retains all the variance of the original data.

    # Obtained output: The variance of the original data matched the eigenvalue sum obtained from PCA.


    """
    current_dir = pca_dir
    file_path = os.path.join(current_dir, "non_zero_columns.csv")
    #  Load all the features from the input file.
    all_features = read_features(file_path)
    
    # Get the scaled features and independent features.
    independent_features, scaled_features = feature_scaling(all_features)
    
    # Get the sorted eigen values and the corresponding eigen vectors.
    sorted_eigval, sorted_eigvec = sorted_eigval_eigvec(scaled_features)
    
    # Get the total variance of the input data
    variance = np.var(scaled_features, axis = 0)
    total_variance = variance.sum()
    
    total_eigen_val = sorted_eigval.sum()
    
    assert(total_variance.round(1) == total_eigen_val.round(1))
    
#===============================================================================================================

# Test function to verify the scaling of features as expected.
def test_feature_scaling():
    """
    
    
    # Purpose of test: To test the functionality of the function feature_scaling, whether it scales the features correctly.

    # Input: A sample numpy array
            all_features = np.array([[4, 2, 7, 4, 6],
                                    [6, 2, 8, 8, 10],
                                    [3, 5, 8, 2, 1]])
    Command to run file: pytest test_pca.py

    # Expected output: An array of Z-score scaled features.
            expected_scaled_features = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
    # Obtained output: Scaled features are obtained by calling the function feature_scaling, whose output should match with the expected ones.
            obtained_scaled_features = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
    """
    
    # Sample array
    all_features = np.array([[4, 2, 7, 4, 6],
                            [6, 2, 8, 8, 10],
                            [3, 5, 8, 2, 1]])
    # Obtaine scaled features by calling the function
    _, obtained_scaled_features = feature_scaling(all_features)
    # Expected scaled features
    expected_scaled_features = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
    
    # Verify if the obtained features match the expected features
    assert(expected_scaled_features - obtained_scaled_features < 1e-3).all()

#===============================================================================================================

def test_sorted_eigval_eigvec():
    """
    
    # Purpose of test: To test the functionality of the function sorted_eigval_eigvec, whether it computes eigen vectors correctly and sort eigen values correctly.

    # Input: A sample numpy array of scaled features.
            scaled_fea = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
            
    Command to run file: pytest test_pca.py
    
    # Expected output: The dot product of eigen vectors should be close to zero and eigen values should be sorted in descending order.

    # Obtained output: Eigen values and eigen vectors are obtained by calling the function sorted_eigval_eigvec, the dot product of eigen vectors is close to zero.

    """
    # Sample numpy array
    scaled_fea = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
    # Obtaining sorted eigen values and eigen vectors by calling the function
    sort_eig_val, sort_eig_vec = sorted_eigval_eigvec(scaled_fea)
    # Verify if the eigen vectors are computed correctly
    assert(np.dot(sort_eig_vec[0], sort_eig_vec[1]) < 1e-3)
    # Verify if the eigen values are sorted as expected
    assert(sort_eig_val[0] > sort_eig_val[1] > sort_eig_val[2])

#===============================================================================================================
# Test function to verify the keisers rule.
def test_keisers_rule():
    """
    
    # Purpose of test: To test the functionality of the function keisers_rule, whether it computes number of principal components based on eigen values correctly.

    # Input: A sample numpy array of sorted eigen values.
            sort_eig_val = np.array([1.5, 1.2, 1.1, 0.9, 0.7])
            
    Command to run file: pytest test_pca.py
    
    # Expected output: The eigen values which are greater than one have to be considered according to keiser's rule.
            expected_num_pc = 3
    # Obtained output: Number of principal components obtained by calling the function keisers_rule should match with the expected value.
            obtained_num_pc = 3
    """
    # Sample eigen values
    sort_eig_val = np.array([1.5, 1.2, 1.1, 0.9, 0.7])
    # Obtained number of principal components by calling the function
    obtained_num_pc = keisers_rule(sort_eig_val)
    # Expected number of principal components
    expected_num_pc = 3
    
    # Verify if the obtained number of principal components matched the expected value.
    assert(expected_num_pc == obtained_num_pc)
    
#===============================================================================================================
# Test function to verify the number of principal components required to explain variance over threshold value.
def test_variance_explained():
    """
    
    # Purpose of test: To test the functionality of the function variance_explained, whether it computes number of principal components required to explain variance over threshold value correctly.

    # Input: A sample numpy array of sorted eigen values and scaled features.
            scaled_fea = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
            eigen_val = np.array([2.74, 0, 0])
            threshold = 0.8
            
    Command to run file: pytest test_pca.py
    
    # Expected output: The eigen values which have more variance has to be considered.
            expected_num_pc = 1
            
    # Obtained output: Number of principal components obtained by calling the function variance_explained should match with the expected value.
            obtained_num_pc = 1
    """
    # Sample scacled features and eigen values
    scaled_fea = np.array([[-0.267, -0.707, -1.414], [ 1.336, -0.707,  0.707], [-1.0690,  1.414,  0.707]])
    eigen_val = np.array([2.74, 0, 0])
    # Obtained number of principal components by calling the function
    threshold = 0.8
    obtained_num_pc = variance_explained(scaled_fea, eigen_val, threshold)
    # Expected number of principal components
    expected_num_pc = 1
    
    # Verify if the obtained number of principal components matched the expected value.
    assert(expected_num_pc == obtained_num_pc)
#===============================================================================================================


def test_pca():
    '''
    # Purpose of test: To test the functionality of the function pca, whether it transforms the dataset from high dimensional to low dimensional correctly.

    # Input: A sample numpy array of eigen vectors, scaled features, and number of principal components.
            scaled_ind_features = np.array([[-0.267, -0.707, -1.414, 2.12], [ 1.336, -0.707,  0.707, 0.42], [-1.0690,  1.414,  0.707, 0.35]])
            sort_eig_vectors = np.ones((4, 4))
            num_pc = 2 
    
    Command to run file: pytest test_pca.py
    
    # Expected output: The shape of the principal components should be (scaled_ind_features.shape[0], num_pc).

    # Obtained output: Principal components obtained by calling the function pca matched with the expected shape.
    '''
    
    # Sample scacled features and eigen vectors
    scaled_ind_features = np.array([[-0.267, -0.707, -1.414, 2.12], [ 1.336, -0.707,  0.707, 0.42], [-1.0690,  1.414,  0.707, 0.35]])
    sort_eig_vectors = np.ones((4, 4))
    num_pc = 2  # Number of principal components
    # Obtained principal components by calling the function
    principal_components = pca(scaled_ind_features, sort_eig_vectors, num_pc)
    # Expected shape of principal components
    expected_pc_shape = (scaled_ind_features.shape[0], num_pc)
    
    # Verify if the obtained shape of principal components matched the expected shape.
    assert(expected_pc_shape == principal_components.shape)

#===============================================================================================================
    
    