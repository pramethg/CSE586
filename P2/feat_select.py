'''
Start code for Project 2
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name:
    PSU Email ID:
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
    selected_inds = np.array([])

    # Select 20 features at random as an example
    selected_inds = np.random.choice(train_data.shape[1], 20, replace=False)
    return selected_inds

# TODO: Implement the filtering method.
def filter_method(train_data, train_labels):
    """
    Performs filter method.
    Args:
        train_data (numpy array): [NxM] vector of training data, where N is the number of samples and M is the number of features.
        train_labels (numpy array): [Nx1] vector of training labels.
    Returns:
        selected_inds (numpy array): a [1xM] vector sorted in descending order of feature importance.
        scores (numpy array): a [1xM] vector containing the scores of the corresponding features.
    """
    # The current method just returns a random selection of features. 
    selected_inds = np.zeros(train_data.shape[1], dtype=np.int32)
    scores = np.zeros(train_data.shape[1])

    selected_inds = np.random.choice(train_data.shape[1], train_data.shape[1], replace=False)
    scores = np.random.uniform(0, 100, train_data.shape[1])
    return selected_inds, scores
    
