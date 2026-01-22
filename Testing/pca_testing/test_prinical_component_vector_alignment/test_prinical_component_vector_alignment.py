
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 04 Feb 2024
Description: Test the alignment of principal vector w.r.t variance based on the known sample datapoints.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pca'))
sys.path.insert(0, pca_dir)
from pca import sorted_eigval_eigvec
np.random.seed(10)

def test_principal_comonent_vector():
    
    """
    Test the principal component vectors alignment of the given data points.
    
    
    # Purpose of test: To make sure that the principal component vectors align as expected with respect to the variance of known sample data points.

    # Input: x (ndarray): The x-coordinates of the data points.
    #   points_1 (ndarray): Y-coordinate of the top set of points.
    #   points_2 (ndarray): Y-coordinate of the medium set of points.
    #   points_3 (ndarray): Y-coordinate of the bottom set of points.
    
    Command to run file: pytest test_prinical_component_vector_alignment.py


    # Expected output: The angle between the principal component vector computed from all the set of points and the angle after removal of some set of points is greater than zero, indicating alignment with the variance of the known sample data points.

    # Obtained output: Angle between the principal component vector computed from all the set of points and the angle after removal of some set of points is greater than zero.


    """
    x = np.linspace(-1, 1, 50)
    # Create y-coordinates points with offset
    points_1 = np.random.uniform(0.5, 2, len(x))
    points_2 = np.random.uniform(0, 0.5, len(x))
    points_3 = np.random.normal(0, 0.5, len(x))
    
    # Generating data points of y-coordinate
    y_inc_top = x + points_1
    y_inc_middle = x + points_2
    y_inc_down = x - points_3
    
    # Stacking the all the points 
    points = np.vstack((np.column_stack((x, y_inc_top)), np.column_stack((x, y_inc_middle)), np.column_stack((x, y_inc_down))))
    # Get eigen values and eigen vectors for all points
    eigen_val, eigen_vec = sorted_eigval_eigvec(points)
    # Compute the angle of the first principal component vector
    angle = np.arctan(eigen_vec[1][0] / eigen_vec[0][0])
    angle = np.degrees(angle) # Get the angle in degrees
    
    # Stacking only middle and bottom set of points
    points_removed = np.vstack((np.column_stack((x, y_inc_middle)), np.column_stack((x, y_inc_down))))
    # Get eigen values and eigen vectors for middel and bottom set of points
    eigen_val_rem, eigen_vec_rem = sorted_eigval_eigvec(points_removed)
    # Compute the angle of the first principal component vector
    angle_rem = np.arctan(eigen_vec_rem[1][0] / eigen_vec_rem[0][0])
    angle_rem = np.degrees(angle_rem)  # Get the angle in degrees
     
    assert(angle - angle_rem > 0)