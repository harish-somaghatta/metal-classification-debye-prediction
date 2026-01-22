
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 14 Feb 2024
Description: Compare the biplot obtained from Principal Component Analysis with the biplot from the website SOGA-py(https://www.geo.fu-berlin.de/en/v/soga-py/Advanced-statistics/Multivariate-Approaches/Principal-Component-Analysis/PCA-the-basics/Interpretation-and-visualization/index.html.)

"""
import matplotlib.pyplot as plt
# Import required functions from pca file.
from pca import read_features, sorted_eigval_eigvec, feature_scaling, keisers_rule, pca

# Read the data which is taken from online resource. (Freie Universität Berlin )
all_features = read_features("C:/Users/shari/Downloads/food-texture.csv")
# Get the independent features and scaled features.
independent_features, scaled_features = feature_scaling(all_features)
# Calculate eigen values and eigen vectors, and sort them in descending order.
sorted_eigval, sorted_eigvec = sorted_eigval_eigvec(scaled_features)
# Get the number of principal components using keisers rule.
num_principal_components = keisers_rule(sorted_eigval)
# Get the principal components.
principal_components = pca(scaled_features, sorted_eigvec, num_principal_components)
# Get the eigen vectors corresponding to the considered principal components.
eigen_vectors = sorted_eigvec[:,:num_principal_components]
# Create a scatter plot of the principa components.
plt.scatter(principal_components[:, 0], principal_components[:, 1], label='Data Points')

# Create lines at x= 0 and y=0 for reference.
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.axvline(x=0, color='green', linestyle='--', linewidth=1)

# Create eigen vectors with the corresponding labels.
plt.quiver(0, 0, eigen_vectors[0, 0], eigen_vectors[0, 1], scale=0.5, color='red', angles='xy', scale_units='xy', label='Oil')
plt.quiver(0, 0, eigen_vectors[1, 0], eigen_vectors[1, 1], scale=0.5, color='blue', angles='xy', scale_units='xy', label='Density')
plt.quiver(0, 0, eigen_vectors[2, 0], eigen_vectors[2, 1], scale=0.5, color='green', angles='xy', scale_units='xy', label='Crispy')
plt.quiver(0, 0, eigen_vectors[3, 0], eigen_vectors[3, 1], scale=0.5, color='violet', angles='xy', scale_units='xy', label='Fracture')
plt.quiver(0, 0, eigen_vectors[4, 0], eigen_vectors[4, 1], scale=0.5, color='orange', angles='xy', scale_units='xy', label='Hardness')

# Give the labels and title
plt.title('Biplot of PC values and variables')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.legend()

# Show the plot
plt.show()
