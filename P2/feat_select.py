'''
Start code for Project 2
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Prameth Gaddale
    PSU Email ID: pqg5273@psu.edu
    Description: (A short description of what each of the functions you've written does.).
}
'''

import numpy as np

# TODO: Implement the sequential forward selection algorithm.
def forward_selection(train_data, train_labels):
    """
    Performs forward selection.
    Args:
        train_data (numpy array): [NxM] vector of training data, where N is the number of samples and M is the number of features.
        train_labels (numpy array): [Nx1] vector of training labels.
    Returns:
        selected_inds (numpy array): a [1xK] vector containing the indices of features
            selected in forward selection. K is the # of feats choosen before selection was terminated.
    """
    # The current method just returns a random selection of features.
    # selected_inds = np.array([])

    # Select 20 features at random as an example
    # selected_inds = np.random.choice(train_data.shape[1], 20, replace=False)
    selected_inds = np.arange(20)
    return selected_inds

# TODO: Implement the filtering method.
def filter_method(train_data, train_labels):
    """
    Performs filter method with Variance Ratio as the evaluation criterion.
    Args:
        train_data (numpy array): [NxM] vector of training data, where N is the number of samples and M is the number of features.
        train_labels (numpy array): [Nx1] vector of training labels.
    Returns:
        selected_inds (numpy array): a [1xM] vector sorted in descending order of feature importance.
        scores (numpy array): a [1xM] vector containing the scores of the corresponding features.
    """
    # Initialize the scores array with zeros.
    scores = np.zeros(train_data.shape[1])
    # Store all the unique training labels
    train_label_unique = np.unique(train_labels)
    # Iterate over all the 1961 features present in the training data
    for feature in range(train_data.shape[1]):
        sum_per_label_var = 0 # variable to store the summation of per label variance
        # Variance of the feature: Var(Sf)
        var_feature = np.var(train_data[:, feature])
        # For each feature iterate over all the labels over each train_label categories.
        for label in train_label_unique:
            # Construct a array represting all the training data representing the label being iterated over.
            train_label_list = train_data[:,feature][train_labels == label]
            # Calculate the variance of the constructed array
            train_label_var = np.var(train_label_list)
            # add the train_label_var to the temp variable representing the sum
            sum_per_label_var+=train_label_var
        # For each feature score divide the summation of per class variance
        complete_sum_label = sum_per_label_var/len(train_label_unique)
        if complete_sum_label!=0:
            scores[feature] = var_feature/complete_sum_label
    # Store the indices of the scores in descending order of magnitude. 
    selected_inds = np.argsort(scores)[::-1]
    return selected_inds, scores