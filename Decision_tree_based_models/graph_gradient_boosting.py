
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("gradient_boosting_hyperpparameter.csv", delimiter = ",", dtype = "float", skip_header = 1)

num_trees = data[:, 0]
min_samples = data[:, 1]
max_depth = data[:, 2]
R2_error = data[:, 3]


def unq_val(parameter, metric):
    
    sorted_metric = (np.argsort(metric)[::-1])
    sorted_parameter = (parameter[sorted_metric])
    
    # Initialize lists to store unique values and their indices
    unique_values = []
    unique_indices = []
    
    # Initialize a set to get unique values
    seen_values = set()
    
    # Iterate over the sorted parameters to find unique values and their indices
    for i, paras in enumerate(sorted_parameter):
        if paras not in seen_values:
            unique_values.append(paras)
            unique_indices.append(i)
            seen_values.add(paras)
    
    R2_values = R2_error[sorted_metric[unique_indices]]
    # Print all unique values and their corresponding indices
    print("Unique values:", unique_values)
    print("Indices of unique values:", R2_values)
    
    sort_unq_val = np.argsort(unique_values)
    
    return np.sort(unique_values), R2_values[sort_unq_val]

# Get the unique values and their corresponding F1 score values
n_trees_list, R2_score_list_trees = unq_val(num_trees, R2_error)
min_samples_list, R2_score_list_min_samples = unq_val(min_samples, R2_error)
max_depth_list, R2_score_list_max_depth = unq_val(max_depth, R2_error)

# Get the optimal values index for scattering in the plot
higest_R2_score_idx_t = np.argmax(R2_score_list_trees)
higest_R2_score_idx_s = np.argmax(R2_score_list_min_samples)
higest_R2_score_idx_d = np.argmax(R2_score_list_max_depth)


# Give figure configuration
fig, ax = plt.subplots(1, 3, figsize=(20, 8))

# Plotting each subplot
ax[0].plot(n_trees_list, R2_score_list_trees, label='number of trees vs coefficient of determination')
ax[0].scatter(n_trees_list[higest_R2_score_idx_t], R2_score_list_trees[higest_R2_score_idx_t], color = 'r', label='Optimal number trees')
ax[0].set_xlabel('Number of Trees')
ax[0].set_ylabel('coefficient of determination')
ax[0].legend()

ax[1].plot(min_samples_list, R2_score_list_min_samples, label='minimum number of samples vs coefficient of determination')
ax[1].scatter(min_samples_list[higest_R2_score_idx_s], R2_score_list_min_samples[higest_R2_score_idx_s], color = 'r', label='Optimal number of samples')

ax[1].set_xlabel('Minimum Samples required to split')
ax[1].set_ylabel('coefficient of determination')
ax[1].legend()

ax[2].plot(max_depth_list, R2_score_list_max_depth, label='max_depth vs coefficient of determination')
ax[2].scatter(max_depth_list[higest_R2_score_idx_d], R2_score_list_max_depth[higest_R2_score_idx_d], color = 'r', label='Optimal depth of a tree')

ax[2].set_xlabel('Maximum Depth of each tree')
ax[2].set_ylabel('coefficient of determination')
ax[2].legend()

plt.show
    
