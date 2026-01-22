
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 28 Apr 2024
Description: Random Forest algorithm for classification of materials into metals and non-metals.

"""

# Import necessary library
import numpy as np
import os
import sys
pca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))
sys.path.insert(0, pca_dir)

ANN_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ANN'))
sys.path.insert(0, ANN_dir)


# Import necessary functions from the files
from ANN_classification import read_dataset, split_dataset
from classification_and_regression_tree import decision_tree, predict_val
from classification_evaluation_metrics import accuracy, precision, recall, f1_score

def bootstrap_samples(x, y, num_samples, seed):
    """
    Creates bootstrap samples from the dataset.

    Parameters:
    x (numpy.ndarray): Independent features of the dataset.
    y (numpy.ndarray): Target feature of the dataset.
    num_samples (int): Number of samples to be randomly selected.

    Returns:
    x[idx] (numpy.ndarray): Subset of independent feature samples.
    y[idx] (numpy.ndarray): Subset of dependent feature samples.
    """
    # Randomly select the subset of samples
    np.random.seed(seed)
    # Index of the subset samples
    idx = np.random.randint(x.shape[0], size=num_samples)
    
    # Return subset of sample space
    return x[idx], y[idx]

def random_feature_subpace(x, n_sub_features, seed):
    """
    Creates a random feature subspace from the given dataset.

    Parameters:
    x (numpy.ndarray): Independent features of the dataset.
    n_sub_features (int): Number of features to be selected for the subspace.

    Returns:
    x[:, random_feature_subspace] (numpy.ndarray): Subset of the given dataset features.
    """
    # Seed the random number to be selected

    np.random.seed(seed)
    # Select the features randomly
    random_feature_subspace = np.random.choice(x.shape[1], size=n_sub_features, replace=True)
    
    # Return random feature subspace from the given full dataset
    return x[:, random_feature_subspace]

def random_forest(x_train, y_train, boot_strapped_samples, num_subfeatures, depth_max, samples_min, num_trees, seed):
    """
    Creates a random forest algorithm.

    Parameters:
    x_train (numpy.ndarray): Independent features of the training dataset.
    y_train (numpy.ndarray): Target feature of the training dataset.
    boot_strapped_samples (int): Number of bootstrapped samples to create.
    num_subfeatures (int): Number of features to select for each tree from the full dataset.
    depth_max (int): Maximum depth of each decision tree.
    samples_min (int): Minimum number of samples needed to split a node.
    num_trees (int): Number of trees in the random forest.

    Returns:
    trees_forest (list): Random forest made by the list of decision trees.
    """
    # Initialize a list to store the decision trees that make up the random forest
    trees_forest = []
    
    # Iterate over the number of trees considered in the forest
    for i in range(num_trees):
        #print(i)
        seed = seed + i

        # Create bootstrapped samples
        x_train_bs, y_train_bs = bootstrap_samples(x_train, y_train, boot_strapped_samples, seed)
        # Create random feature subspace
        x_train_bs_fs = random_feature_subpace(x_train_bs, num_subfeatures, seed)
        
        # Create a decision tree based on the bootstrapped samples and random feature subspace
        tree = decision_tree(x_train_bs_fs, y_train_bs,depth_max , "classification", samples_min)
        #print(tree)
        # Append the decision tree into the list of trees that make up the forest
        trees_forest.append(tree)
    
    # Return list of decision trees that make up the forest
    return trees_forest

def predict_random_forest(forest, x_test):
    """
    Predicts the class label using a random forest algorithm.

    Parameters:
    forest (list): List of decision trees that make up the random forest.
    x_test (numpy.ndarray): Independent features of the test dataset.

    Returns:
    final_y_pred (list): List of predicted class labels.
    """
    
    # Initialize a list to store predicted class labels
    final_y_pred = []
    # Initialize a list to store prediction arrangements
    arrange_pred = []
    # Initialize a list to store predictions of each tree
    y_prediction = []
    
    # Iterate over trees in the forest
    for tree in forest:
        # Predict the class label based on the current tree 
        y_tree_pred = predict_val(x_test, tree)
        y_prediction.append(y_tree_pred)
        
    #print(len(y_prediction))
    
    # Iterate over number of trees
    for i in range(len(y_prediction[0])):
        arrange_pred.append([])
    # Arrange the predictions suitable for random forest classification
    for sublist in y_prediction:
        for i, val in enumerate(sublist):
            arrange_pred[i].append(val)
    
    # Predict the class label based on the mode of the class label
    for pred_list in arrange_pred:
        y_pred_class, y_pred_count = np.unique(pred_list, return_counts=True)
        y_pred_i = y_pred_class[np.argmax(y_pred_count)]
        final_y_pred.append(y_pred_i)
    
    # Return the finally predicted class labels
    return final_y_pred

def calculate_evaluation_metrics(prediction, true_label):
    """
    Compute evaluation metrics for classification.
    
    Parameters:
    prediction (numpy array): Predicted labels for test dataset.
    true_label (numpy array): True labels for test dataset.
    
    Returns:
    test_accuracy (float): Accuracy of the test dataset.
    test_precision (float): Precision of the test dataset.
    test_recall (float): Recall of the test dataset.
    test_f1_score (float): F1-score of the test dataset.
    """
   
    try:
        # Compute accuracy of the test dataset
        test_accuracy = accuracy(prediction, true_label)
        print("Test accuracy: ", test_accuracy)
        
        # Compute precision of the test dataset
        test_precision = precision(prediction, true_label)
        print("Test precision: ", test_precision)
        
        # # Compute recall of the test dataset
        test_recall = recall(prediction, true_label)
        print("Test recall: ", test_recall)
        
        ## Compute F1-score of the test dataset
        test_f1_score = f1_score(test_precision, test_recall)
        print("Test f1_score: ", test_f1_score)
    except ZeroDivisionError:
        test_accuracy, test_precision, test_recall, test_f1_score = 0, 0, 0, 0
        
    return test_accuracy, test_precision, test_recall, test_f1_score

def main_random_forest(x_train, x_test, y_train, y_test, n_trees, max_depth, min_samples, seed):
    """
    Train a random forest model on the given dataset and evaluate its performance.

    Parameters:
    n_trees (int): Number of decision trees in the random forest.
    max_depth (int): Maximum depth of each decision tree in the random forest.
    min_samples (int): Minimum number of samples required to split a node in each decision tree.
    file (str): Path to the dataset file.
    seed (int): Seed for random number generator.

    Returns:
    y_hat (numpy.ndarray): Predicted target values on the test set.
    test_f1_score (float): F1 score of the model on the test set.
    """
    
    
    # Get the boot strapping samples
    n_samples_bs = int((x_train.shape[0])/2)
    if x_train.shape[1] >2 :
        n_sub_features = int(np.sqrt(x_train.shape[1]))
    else:
        n_sub_features = (x_train.shape[1])
    #print(n_trees)
    forest_trees = random_forest(x_train, y_train, n_samples_bs, n_sub_features, max_depth, min_samples, n_trees, seed )
    
    y_hat = predict_random_forest(forest_trees, x_test)
    #print(len(y_hat))
    y_hat = np.array(y_hat)
    
    test_accuracy, test_precision, test_recall, test_f1_score = calculate_evaluation_metrics(y_hat, y_test)
    
    return y_hat, test_f1_score
    
if __name__ == "__main__":
    num_trees = 7
    maximum_depth = 20
    minimum_samples = 0
    seed = 10
    file_path = os.path.join(pca_dir, 'principal_components_with_dependent_features.csv')
    # Read the dataset and split into train and test data
    independent_features, target_feature = read_dataset(file_path)
    x_train, x_test, y_train, y_test,_ = split_dataset(independent_features, target_feature, train_ratio = 0.8 )
    y_prediction, f1_score = main_random_forest(x_train, x_test, y_train, y_test, num_trees, maximum_depth, minimum_samples, seed)