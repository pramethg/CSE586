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

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def arg_parse():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter_feat_count', type=int, default=100, help='Number of features to be selected by filter method.')
    parser.add_argument('--num_vis_feat', type=int, default=10, help='Number of features to be visualized.')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training')
    parser.add_argument('--save_results', action='store_true', help='Whether to save results')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--do_plot', action='store_true', help='Whether to plot results')
    return parser.parse_args()

def load_dataset(data_path='data'):
    """
    Loads the dataset.
    """
    data = np.load(os.path.join(data_path, 'data.npz'))

    # Remove NaNs
    clean_data = data['data']
    clean_data[np.isnan(clean_data)] = 0.

    return clean_data, data['labels'], data['sub_info'], data['form_names'], data['feat_names']


def normalize_data(data, min_value=None, max_value=None):
    """
    Normalizes the data to [0, 1] range. Records the min and max values if 
    sent training data. If sent test data, uses the previous min and max values.
    Args:
        data (numpy array): Data to be normalized.
        min_value (float): Minimum value of the data.
        max_value (float): Maximum value of the data.
    Returns:
        data_norm: Normalized data.
        min: Minimum value of the data.
        max: Maximum value of the data.
    """
    if min_value is None:
        min_value = np.min(data)
    if max_value is None:
        max_value = np.max(data)

    data_norm = (data - min_value) / (max_value - min_value)
    return data_norm, min_value, max_value

def split_data(data, label, sub_info, i):
    """
    Splits the data in a LOSO fashion with the i-th subject as test data.
    Args:
        data (numpy array): [NxM] vector of data to be split, where N is the number of samples and M is the number of features.
        label (numpy array): [N] vector of labels of the data.
        sub_info (numpy array): [Nx2] vector of subject information of the data where N is the number of samples.
        i (int): Test subject.
    Returns:
        train_data: Training data.
        train_label: Training labels.
        test_data: Test data.
        test_label: Test labels.
    """
    # Get the indices of the test and train data
    test_inds = np.where(sub_info[:, 0] == i)[0]
    train_inds = np.where(sub_info[:, 0] != i)[0]

    # Get the train and test data
    train_data = data[train_inds, :]
    train_label = label[train_inds]
    test_data = data[test_inds, :]
    test_label = label[test_inds]
    return train_data, train_label, test_data, test_label

def plot_feat(feat_stat, feat_name, num_on_bar, title, xlabel, filename, save_dir='results'):
    """
    Plots the feature statistics.
    Args:
        feat_stat ([nx2] numpy array): Feature statistics where n is the number of features and 
            the second dim. is feature number and score.
        feat_name (list): all feature names.
        num_on_bar (int): Number of features to be shown on the bar.
        title (str): Title.
        xlabel (str): X-axis label.
        filename (str): Output filename.
    """
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    if feat_stat.shape[0] < num_on_bar:
        num_on_bar = feat_stat.shape[0]
    
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_on_bar), feat_stat[:num_on_bar, 1])
    ax.set_xticks(np.arange(num_on_bar))
    ax.set_xticklabels(feat_stat[:num_on_bar, 0], rotation=90)
    ax.set_yticks(np.arange(0, num_on_bar, 1))
    ax.set_yticklabels(feat_name[:num_on_bar], rotation=0)
    ax.set_ylabel('Features')
    ax.set_ylim([0.5, num_on_bar + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plots', 'feature_stats.png'))
    plt.close(fig)

def sub_stats(sub_pred, labels, num_forms):
    """
    Calculates the subject statistics.
    Args:
        sub_pred (numpy array): Subject predictions.
        labels (numpy array): Labels.
        num_forms (int): Number of forms.
    """
    sub_class_total = np.zeros(num_forms)
    sub_class_correct = np.zeros(num_forms)

    # Get subject per class accuracy
    for i in range(len(sub_pred) ):
        pred = sub_pred[i]
        label = labels[i]
        sub_class_total[label] += 1
        if pred == label:
            sub_class_correct[label] += 1

    

    sub_classes = sub_class_correct / sub_class_total
    sub_classes[np.isnan(sub_classes)] = 0.

    # Get conf matrix
    conf_mat = confusion_matrix(labels, sub_pred)
    conf_mat[np.isnan(conf_mat)] = 0.
    conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)

    return sub_classes, conf_mat