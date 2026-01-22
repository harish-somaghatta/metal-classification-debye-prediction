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
from pca import sorted_eigval_eigvec
np.random.seed(10)

def test_principal_comonent_vector(x, points_1, points_2, points_3):
    """
    Test the principal component vectors alignment of the given data points.

    Parameters:
    x (ndarray): The x-coordinates of the data points.
    points_1 (ndarray): Y-coordinate of the top set of points.
    points_2 (ndarray): Y-coordinate of the medium set of points.
    points_3 (ndarray): Y-coordinate of the bottom set of points.

    Returns:
    None
    """
    
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
     
    # Generate a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plotting all points with principal components vectors
    axs[0].scatter(points[:, 0], points[:, 1])
    axs[0].quiver(0, 0, eigen_vec[0, 0], eigen_vec[1, 0], color="red", scale=3, label='First principal component')
    axs[0].quiver(0, 0, eigen_vec[0, 1], eigen_vec[1, 1], color="green", scale=3, label='Second principal component')
    axs[0].axhline(0, color='blue', linestyle='-')
    axs[0].axvline(0, color='blue', linestyle='-')
    axs[0].set_title('PC vector alignment (All Points)')
    axs[0].set_xlabel('X-coordinate points')
    axs[0].set_ylabel('Y-coordinate points')
    axs[0].axis('equal')
    axs[0].legend(loc='upper left')
    axs[0].text(0.5, -2.5, f'Angle: {angle:.2f} degrees', fontsize=10)
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-2, 2)
    
    # Plotting only middle and bottom set of points with principal components vectors
    axs[1].scatter(points_removed[:, 0], points_removed[:, 1])
    axs[1].quiver(0, 0, eigen_vec_rem[0, 0], eigen_vec_rem[1, 0], color="red", scale=3, label='First principal component')
    axs[1].quiver(0, 0, eigen_vec_rem[0, 1], eigen_vec_rem[1, 1], color="green", scale=3, label='Second principal component')
    axs[1].axhline(0, color='blue', linestyle='-')
    axs[1].axvline(0, color='blue', linestyle='-')
    axs[1].set_title('PC vector alignment (less points)')
    axs[1].set_xlabel('X-coordinate points')
    axs[1].set_ylabel('Y-coordinate points')
    axs[1].axis('equal')
    axs[1].legend(loc='upper left') 
    axs[1].text(0.3, -1.6, f'Angle: {angle_rem:.2f} degrees', fontsize=10)
    axs[1].set_xlim(-3, 3)
    axs[1].set_ylim(-2, 2)
    # Showing the plot
    plt.show()
    

# Create x-coordinate points
x = np.linspace(-1, 1, 50)
# Create y-coordinates points with offset
random_values_top = np.random.uniform(0.5, 2, len(x))
random_values_middle = np.random.uniform(0, 0.5, len(x))
random_values_bottom = np.random.normal(0, 0.5, len(x))

# Test the alignment of the principal component vectors
test_principal_comonent_vector(x, random_values_top, random_values_middle, random_values_bottom)

