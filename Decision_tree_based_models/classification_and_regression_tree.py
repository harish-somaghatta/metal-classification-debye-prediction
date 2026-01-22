
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 18 Apr 2024
Description: CART(Classification and Regression Trees) for Random Forest classification and Gradient Boosting regression.

"""

# Import the necessary library
import numpy as np

def tree_leaf(target_feature, model_type):
    """
    To find the classification label or prediction value for a leaf node.
    
    Parameters:
    target_feature (numpy.array): The target labels of the leaf node.
    model_type (str): Model type, 'classification' or 'regression'.
    
    Returns:
    label (float): The classification label or prediction value for a leaf node.
    """
    
    if model_type == "classification":
        # Get the unique classes and their counts
        unq_classes, class_count = np.unique(target_feature, return_counts = True)
        # Get the class with maximum count
        max_count_class = unq_classes[np.argmax(class_count)]
        
        return max_count_class
    
    elif model_type == "regression":
        # Compute the mean of the target feature
        reg_label = np.mean(target_feature)
        
        return reg_label
        
def calculate_impurity(label_data, model_type):
    
    """
    Calculate the Gini impurity for classification, Mean squared error for regression.

    Parameters:
    label_data (numpy.array): Target feature labels/values .
    model_type (str): Model type, 'classification' or 'regression'.

    Returns:
    impurity (float): Value of impurity.
    
    """

    if model_type == "classification":
        # Get the unique classes and their counts
        unq_labels, label_count = np.unique(label_data, return_counts = True)
        # Get the probability
        probability = label_count / label_data.size
        sqr_prob = probability ** 2
        total_sqrd_prob = np.sum(sqr_prob)
        # Gini impurity
        gini_impurity = 1 - total_sqrd_prob

        return gini_impurity
    
    elif model_type == "regression":
        # Find the mean of the target feature values.
        target_mean = np.mean(label_data)
        mean_diff = (label_data - target_mean) ** 2
        
        # Return mean squared error value
        return np.mean(mean_diff)
        
def best_values(independent_features, target_feature, model_type):
    """
    Find the best split value, best feature index and best impurity gain value.

    Parameters:
    independent_features (numpy.array): Independent features.
    target_feature (numpy.array): Target feature labels/values.
    model_type (str): Model type, 'classification' or 'regression'.

    Returns:
    best_split_val (float): Best split value of the feature.
    best_feature_idx (int): Best feature index.
    best_imp_gain (float): Impurity gain of the best split.

    """
    # Get the number of samples
    num_samples = target_feature.shape[0]
    
    num_indenpendent_features = independent_features.shape[1]
    # Initialize the best split value, best feature index and impurity gain of best split
    best_feature_idx, best_split_val, best_imp_gain = -1, 1e10, 0
    # Calculate the impurity of the cuurent node
    node_impurity = calculate_impurity(target_feature, model_type)

    # Iterate over each feature for potential splits
    for feature in range(num_indenpendent_features):
        # Extract the values of the current feature
        feature_values = independent_features[:, feature]
        # Find unique values of the feature
        unq_feature_values = np.unique(feature_values)
        
        # Iterate over potential split values
        for idx in range(1, len(unq_feature_values)):
            
            # Calculate the potential split value as the average of adjacent unique feature values
            potential_split = (unq_feature_values[idx] + unq_feature_values[idx-1])/2
            
            # Split the target feature based on the potential split value
            left_node_t = target_feature[feature_values <= potential_split]
            right_node_t = target_feature[feature_values > potential_split]
            
            # Calculate the probabilities of samples in the left and right nodes
            prob_l_node = left_node_t.size / num_samples
            prob_r_node = right_node_t.size / num_samples
            
            # Calculate the impurity of the left and right nodes
            left_impurity = calculate_impurity(left_node_t, model_type)
            right_impurity = calculate_impurity(right_node_t, model_type)
            
            # Calculate the gini gain for the potential split            
            gini_gain = node_impurity - ((prob_r_node * right_impurity) + (prob_l_node * left_impurity))
            
            # If the impurity gain is greater, update the optimal split parameters.
            if gini_gain > best_imp_gain:
                best_split_val, best_feature_idx = potential_split, feature
                best_imp_gain = gini_gain

    return best_split_val, best_feature_idx, best_imp_gain


def split_dataset_(independent_features, target_feature, best_split_val, best_feature_idx, model_type):
    """
    Split the dataset into left and right nodes based on the best split value and best feature index.
    
    Parameters:
    independent_features (numpy.ndarray): Independent features of the dataset.
    target_feature (numpy.ndarray): Target feature of the dataset.
    best_split_val (float): Best split value found during tree construction.
    best_feature_idx (int): Best feature index for splitting.
    model_type (str): Model type, 'classification' or 'regression'.
    
    Returns:
    left_node_fea (numpy.ndarray): Independent features of the left node after splitting.
    right_node_fea (numpy.ndarray): Independent features of the right node after splitting.
    left_node_tar (numpy.ndarray): Target feature of the left node after splitting.
    right_node_tar (numpy.ndarray): Target feature of the right node after splitting.
    
    """
    
    # Extract the values of the best feature for splitting
    feature_values = independent_features[:, best_feature_idx]
    
    # Split the independent features based on the best split value and best feature index
    left_node_fea = independent_features[feature_values <= best_split_val]
    right_node_fea = independent_features[feature_values > best_split_val]
    
    # Split the target feature based on the best split value and best feature index
    left_node_tar = target_feature[feature_values <= best_split_val]
    right_node_tar = target_feature[feature_values > best_split_val]
    
    return left_node_fea, right_node_fea, left_node_tar, right_node_tar


def decision_tree(independent_features, target_feature, max_depth, model_type, min_samples):
    """
    Builds a decision tree recursively based on the given dataset.

    Parameters:
    independent_features (numpy.ndarray): Independent features of the dataset.
    target_feature (numpy.ndarray): Target feature of the dataset.
    max_depth (int): Maximum depth of the decision tree.
    model_type (str): Model type, 'classification' or 'regression'.
    min_samples (int): Minimum number of samples required to perform a split.

    Returns:
    (feature_idx, split_val , left_tree, right_tree) (tuple) : A tuple containing feature index, split value, left node tree, right node tree.

    """
    # Get the unique classes/values of the target feature
    unique_classes = np.unique(target_feature)
    
    # Check until the given conditions are met
    if len(unique_classes) == 1 or max_depth == 0 or len(target_feature) < min_samples:
        return tree_leaf(target_feature, model_type)
    
    else:
        
        # Find the best split value, best feature index, and impurity of the best split
        split_val, feature_idx, impurity = best_values(independent_features, target_feature, model_type)
        # Split the dataset into left and right nodes
        left_node, right_node, left_target, right_target = split_dataset_(independent_features, target_feature, split_val, feature_idx, model_type)
        #print((left_node, right_node, left_target, right_target))
        #print(f"Now left Node, depth is {max_depth - 1}")
        # Recursively construct the left and right subtrees
        left_tree = decision_tree(left_node, left_target, max_depth - 1, model_type, min_samples)
        #print(f"Now right Node, depth is {max_depth - 1}")
        right_tree = decision_tree(right_node, right_target, max_depth - 1 , model_type, min_samples)
        
        return (feature_idx, split_val , left_tree, right_tree)

def classify_example(example, tree):
    """
    Classifies an example based on the built decision tree.

    Parameters:
    example (numpy.ndarray): Example to be classified.
    tree (tuple): Decision tree node or leaf node value.

    Returns:
    tree or classify_example( tuple, float): Returns class label or prediction value.

    """
    
    if isinstance(tree, tuple):
        # If the tree is a tuple, it's a decision node
        feature_idx, split_val, left_tree, right_tree = tree
        
        # Extract the feature value for the example
        example_value = example[feature_idx]
        
        # Based on the split value, choose left node or right node
        if example_value <= split_val:
            return classify_example(example, left_tree)
        else:
            return classify_example(example, right_tree)
    else:
        # Return class label or value, if the tree is leaf node
        return tree
    
def predict_val(x, tree):
    """
    Using the built decision tree, predict the class label or value.

    Parameters:
    x (numpy.ndarray): Examples to predict.
    tree (tuple): Decision tree used to predict.

    Returns:
    y_pred (list): Predicted target labels or values for the given examples.
    
    """
    
    # Initialize a list to store the predicted class label/values
    y_pred = []
    
    # Iterate over each example
    for example in x:
        # Classify the each example using the decision tree
        predicted_label = classify_example(example, tree)
        y_pred.append(predicted_label)
        
    return y_pred
        
