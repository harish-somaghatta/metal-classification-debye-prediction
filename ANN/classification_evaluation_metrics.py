"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 28 Feb 2024
Description: Evaluation metrics for binary classification models.

"""

def accuracy(prediction, true_label):
    """
    Calculates the accuracy of the model.

    Args:
    y_true (numpy array): True labels.
    y_pred (numpy array): Predicted labels.

    Returns:
    accuracy_val (float): Accuracy of the model.
    """
    
    true_predictions = 0 # Initialize true predictions
    n_predictions = len(true_label)  # Total number of predictions
    
    # Iterate over predictions and true labels
    for pred_val, true_val in zip(prediction, true_label):
        # Check whether the predicted values and true values are equal 
        if pred_val == true_val:
            true_predictions += 1
    #print(true_predictions)
    
    accuracy_val = (true_predictions / n_predictions) * 100
    
    return accuracy_val



def precision(prediction, true_label):
    """
    Calculates the precision of the model.

    Args:
    y_true (numpy array): True labels.
    y_pred (numpy array): Predicted values.

    Returns:
    precision_val (float): Precision of the model.
    """
    
    true_positives, false_positives  = 0, 0 # Initialize true predictions  
    
    # Iterate over predictions and true labels
    for pred_val, true_val in zip(prediction, true_label):
        
        if pred_val == 1:
            # Check whether the predicted values and true values are 1.
            if true_val == 1:
                true_positives += 1
            # Check whether the predicted values is 1 and true values is 0.
            else:
                false_positives +=1
              
    #print(true_positives, false_positives)
    precision_val = true_positives / (true_positives + false_positives)
    
    return precision_val

def recall(prediction, true_label):
    """
    Calculates the recall of the model.

    Args:
    y_true (numpy array): True labels.
    y_pred (numpy array): Predicted values.

    Returns:
    precision (float): recall of the model.
    """
    
    true_positives, false_negatives  = 0, 0 # Initialize true predictions  
    
    # Iterate over predictions and true labels
    for pred_val, true_val in zip(prediction, true_label):
        # Check whether the predicted values and true values are equal 
        if pred_val == 1:
            # Check whether the predicted values and true values are 1.
            if true_val == 1:
                true_positives += 1
        # Check whether the predicted value is 0 and true value is 1.
        elif pred_val == 0:
            
            if true_val == 1:
                false_negatives +=1
              
    recall_val = true_positives / (true_positives + false_negatives)
    
    return recall_val

def f1_score(precision, recall):
    """
    Calculates the f1_score of the model.

    Args:
    y_true (numpy array): True labels.
    y_pred (numpy array): Predicted values.

    Returns:
    precision (float): f1_score of the model.
    """
    
    f1_score = 2 * ((precision * recall)/(precision + recall))
    
    return f1_score
    
    
    
    


