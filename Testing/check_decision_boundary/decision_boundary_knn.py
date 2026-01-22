# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Social_Network_Ads.csv'))
sys.path.insert(0, db_file)
KNN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'knn'))
sys.path.insert(0, KNN_dir)

from knn_classification import main_knn

def decision_boundary_generation_knn(x_train, x_test, y_train, y_test, xx, yy, k):
    
    """
    Generate decision boundary for KNN classification.

    Parameters:
    x_train (numpy.ndarray): Training features.
    x_test (numpy.ndarray): Testing features.
    y_train (numpy.ndarray): Training labels.
    y_test (numpy.ndarray): Testing labels.
    xx (numpy.ndarray): Meshgrid for x-axis.
    yy (numpy.ndarray): Meshgrid for y-axis.
    k (int)          : Number of neighbors to consider.

    Returns:
    None 
    """
    
    Z, _ = main_knn(x_train, grid_points, y_train, y_test = None, k = k)
    Z = np.array(Z) 
    Z =  np.reshape(Z, xx.shape)
    # Plot the decision boundary
    
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.flatten(), edgecolors='k', marker='o', label='Train')

    # Plot the testing points
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.flatten(), edgecolors='k', marker='x', label='Test')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary for KNN Classification')
    plt.legend()
    plt.show()

    return None

all_features = np.genfromtxt(db_file, delimiter = ",", dtype = "float", skip_header = 1)

# Extract independent features (assuming they are in columns 2 and 3)
independent_features = all_features[:, [2, 3]]
# Extract dependent feature (assuming it is in column 4)
dependent_feature = all_features[:, 4]

# Number of samples
n_samples = independent_features.shape[0]

# Split the dataset into training and testing sets
train_ratio = 0.8
split_idx = int(n_samples * train_ratio)

x_train, x_test = np.split(independent_features, [split_idx])
y_train, y_test = np.split(dependent_feature, [split_idx])

# Reshape y_train and y_test to be 2D arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Give the decision boundary conditions
x_min, x_max = independent_features[:, 0].min() - 1, independent_features[:, 0].max() + 1

y_min, y_max = independent_features[:, 1].min() - 1, independent_features[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1000))

grid_points = np.hstack((xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)))

decision_boundary_generation_knn(x_train, x_test, y_train, y_test, xx, yy, k=3)