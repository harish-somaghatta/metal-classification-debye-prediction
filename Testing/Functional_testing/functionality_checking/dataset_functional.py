# Functional test required datasets

import numpy as np
import matplotlib.pyplot as plt

def regression_dataset():
    """
    Create sample regression datasets for testing.

    Returns:
    x_train (numpy.ndarray): Independent features for training, an array of shape (49, 2).
    y_train (numpy.ndarray): Dependent features for training, an array of shape (49, 1).
    x_test (numpy.ndarray): Independent features for testing, an array of shape (4, 2).
    y_test (numpy.ndarray): Expected values for test datapoints, an array of shape (4, 1).
    """
    # Training dataset: independent features
    x_train = np.array([[i, 2] for i in range(1, 50)])
    # Training dataset: dependent features
    y_train = np.array([[i * 2 ] for i in range(1, 50)])
    
    # Test dataset: independent features
    x_test = np.array([[20, 2], [15, 2], [11, 2], [5, 2]])
    # Calculate expected values for test datapoints
    y_test = 2 * x_test[:,0].reshape(-1, 1)
    
    return x_train, y_train, x_test, y_test


def classification_dataset():
    """
    Crate a sample classification dataset for testing.

    Returns:
    x_train (numpy.ndarray): Independent features for training, an array of shape (8, 2).
    y_train (numpy.ndarray): Dependent features for training, an array of shape (8, 1).
    x_test (numpy.ndarray): Independent features for testing, an array of shape (8, 2).
    y_test (numpy.ndarray): Dependent features for testing, an array of shape (8, 1).
    """
    # Give the slope (m) and y-intercept (c) for the linear equation y = mx + c
    m, c = 2, 3
    # Create an array of x values ranging from 0 to 24
    x_vals = np.arange(0, 25, 2)
    # Compute the corresponding y values for the linear equation
    y_vals = m * x_vals + c
    
   
    # Give specific x values that will be used to create data points above and below the line
    x_points_above = np.array([3, 14, 10, 7, 12, 20, 18, 5])
    y_points_above = m * x_points_above + c + 3
    y_points_below = m * x_points_above + c - 3
    x_above_points = (np.array([x_points_above, y_points_above])).T
    x_below_points = (np.array([x_points_above, y_points_below])).T
    
    
    y_above_points = np.ones((len(x_above_points), 1))
    y_below_points = np.zeros((len(x_below_points), 1))             

    # Generate the training dataset and test dataset by selecting specific points from the above and below points
    x_train = np.vstack((x_above_points[:2, :], x_below_points[:2, :], x_above_points[4:6, :], x_below_points[4:6,:]))
    x_test = np.vstack((x_above_points[2:4, :], x_below_points[2:4, :], x_above_points[6:8, :], x_below_points[6:8, :]))
    
    y_train = np.vstack((y_above_points[:2, :], y_below_points[:2, :], y_above_points[4:6, :], y_below_points[4:6,:]))
    y_test = np.vstack((y_above_points[2:4, :], y_below_points[2:4, :], y_above_points[6:8, :], y_below_points[6:8, :]))
    
    # Plot the linear equation
    #plt.plot(x_vals, y_vals, label = f'{m}.x + {c}')
    #plt.legend()
    
    # Plot the points above and below the line
    #plt.scatter(x_above_points[:,0], x_above_points[:,1], color = 'r', label = "labeled as 1's")
    #plt.scatter(x_below_points[:,0], x_below_points[:,1], color = 'g', label = "labeled as 0's")
    #plt.legend()
    
    # Add labels and title
    #plt.xlabel('Feature 1')
    #plt.ylabel('Feature 2')
    #plt.title('Classification Dataset')
    
    # Show the plot
    #plt.show()
    #print(y_test)
    return x_train, y_train, x_test, y_test

#regression_dataset()
classification_dataset()