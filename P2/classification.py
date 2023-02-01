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

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import ConfusionMatrixDisplay

from feat_select import *
from util import *


def train(args):
    """
    Perform the feature selection
    """
    data, labels, sub_info, form_names, feat_names  = load_dataset()
    num_subs = len(np.unique(sub_info[:, 0])) + 1
    num_feats = data.shape[1]
    num_forms = len(form_names) + 1
    filter_feat_count = args.filter_feat_count

    sub_train_acc = np.zeros(num_subs)
    sub_class_train = np.zeros((num_subs, num_forms))
    sub_test_acc = np.zeros(num_subs)
    sub_class_test = np.zeros((num_subs, num_forms))
    overall_train_mat = np.zeros((num_forms, 1))
    overall_test_mat = np.zeros((num_forms, 1))

    if args.save_results:
        if not os.path.exists(os.path.join(args.save_dir, 'stats')):
            os.makedirs(os.path.join(args.save_dir, 'stats'))

    for i in range(1, num_subs):
        print(f'Starting training and evaluation for subject {i}...')
        start = time.time()

        # Split and normalize data
        train_data, train_label, test_data, test_label = split_data(data, labels, sub_info, i)

        # TODO: Add data normalization here. Implement the code in the normalize_data function in util.py 


        # Perform feature filtering 
        if num_feats < filter_feat_count:
            filter_feat_count = num_feats

        # TODO: Implement the filter method. 
        filter_inds, filter_scores = filter_method(train_data, train_label)
        train_data = train_data[:, filter_inds[:filter_feat_count]]
        test_data = test_data[:, filter_inds[:filter_feat_count]]

        # TODO: Implement the forward selection method.
        # Perform feature selection
        selected_inds = forward_selection(train_data, train_label)
        train_data = train_data[:, selected_inds]
        test_data = test_data[:, selected_inds]

        # Train and evaluate the model
        model = KNN(n_neighbors=5) # You are free to use any classifier/classifier configuration
        model.labels_ = np.unique(train_label)
        model.fit(train_data, train_label)

        sub_pred_train = model.predict(train_data)
        sub_pred_test = model.predict(test_data)

        # Get model evaluation metrics
        train_acc = np.sum(sub_pred_train == train_label) / len(train_label)
        sub_train_acc[i] = train_acc
        test_acc = np.sum(sub_pred_test == test_label) / len(test_label)
        sub_test_acc[i] = test_acc

        # Get subject-wise statistics
        sub_classes_train, sub_conf_mat_train = sub_stats(sub_pred_train, train_label, num_forms)
        sub_classes_test, sub_conf_mat_test = sub_stats(sub_pred_test, test_label, num_forms)
        sub_class_train[i, :] = sub_classes_train
        sub_class_test[i, :] = sub_classes_test

        # Get overall statistics
        overall_train_mat = overall_train_mat + (1/num_subs) * sub_conf_mat_train
        overall_test_mat = overall_test_mat + (1/num_subs) * sub_conf_mat_test

        # Print results
        print(f'Training accuracy for subject {i}: {train_acc}')
        print(f'Testing accuracy for subject {i}: {test_acc}')

        # Save results
        if args.save_results:
            sub_file = os.path.join(args.save_dir, 'stats', f'subject_{i}.npz')
            np.savez(sub_file, filter_feat_count=filter_feat_count, filter_inds=filter_inds, filter_scores=filter_scores,  
                selected_inds=selected_inds, train_acc=train_acc, test_acc=test_acc, sub_classes_train=sub_classes_train,
                sub_classes_test=sub_classes_test, sub_conf_mat_train=sub_conf_mat_train, sub_conf_mat_test=sub_conf_mat_test)

        print(f"Time taken for subject {i}: {time.time() - start} seconds")
        print('------------------------------------')

    # overall statistics
    overall_train_acc = np.mean(sub_train_acc)
    overall_per_class_train = np.mean(sub_class_train, axis=0)
    overall_test_acc = np.mean(sub_test_acc)
    overall_per_class_test = np.mean(sub_class_test, axis=0)

    print(f'Overall training accuracy: {overall_train_acc}')
    print(f'Overall testing accuracy: {overall_test_acc}')

    if args.save_results:
        overall_file = os.path.join(args.save_dir, 'stats', 'overall.npz')
        np.savez(overall_file, overall_train_acc=overall_train_acc, overall_per_class_train=overall_per_class_train,
            overall_test_acc=overall_test_acc, overall_per_class_test=overall_per_class_test, overall_train_mat=overall_train_mat,
            overall_test_mat=overall_test_mat, sub_train_acc=sub_train_acc, sub_test_acc=sub_test_acc, sub_class_train=sub_class_train,
            sub_class_test=sub_class_test)

    return


# Feel free to edit/add to this function in any way you see fit. This provides the minimum
# functionality required for the assignment.
def visualize(args):
    """
    Visualize the results
    """
    data, labels, sub_info, form_names, feat_names = load_dataset()
    num_subs = len(np.unique(sub_info[:, 0])) + 1
    num_feats = data.shape[1]
    num_forms = len(form_names) + 1
    form_names = np.insert(form_names, 0, 'Transition')
    form_names = np.arange(num_forms)

    # Load results
    overall_file = os.path.join(args.save_dir, 'stats', 'overall.npz')
    overall_results = np.load(overall_file)

    if not os.path.exists(os.path.join(args.save_dir, 'plots')):
        os.makedirs(os.path.join(args.save_dir, 'plots'))

    # Overall subject training and testing rates as a grouped bar chart
    sub_train_acc = overall_results['sub_train_acc']
    sub_test_acc = overall_results['sub_test_acc']
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_subs), sub_train_acc, width=0.35, label='Training')
    ax.bar(np.arange(num_subs)+0.35, sub_test_acc, width=0.35, label='Testing')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Subject')
    ax.set_title('Subject-wise training and testing accuracies')
    ax.set_xticks(np.arange(1, num_subs)+0.35/2)
    ax.set_xticklabels(np.arange(1, num_subs - 0.35, dtype=int))
    ax.set_xlim([0.5, num_subs ])
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'subject_wise_acc.png'))
    plt.close()


    # Overall per class training data. Tilt the x-axis labels by 45 degrees
    overall_per_class_train = overall_results['overall_per_class_train']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(num_forms), overall_per_class_train)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Form')
    ax.set_title('Overall per class training accuracy')
    ax.set_xticks(np.arange(num_forms))
    ax.set_xticklabels(form_names, rotation='vertical')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_per_class_train.png'))
    plt.close()

    # Overall per class testing data
    overall_per_class_test = overall_results['overall_per_class_test']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(num_forms), overall_per_class_test)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Form')
    ax.set_title('Overall per class testing accuracy')
    ax.set_xticks(np.arange(num_forms))
    ax.set_xticklabels(form_names, rotation='vertical')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_per_class_test.png'))

    # Overall training confusion matrix with sklearns display
    fig, ax = plt.subplots(figsize=(10, 10))
    overall_train_mat = overall_results['overall_train_mat']
    disp = ConfusionMatrixDisplay(overall_train_mat, display_labels=form_names)
    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
    ax.set_title('Overall training confusion matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_train_conf_mat.png'))

    # Overall testing confusion matrix with sklearns display
    fig, ax = plt.subplots(figsize=(10, 10))
    overall_test_mat = overall_results['overall_test_mat']
    disp = ConfusionMatrixDisplay(overall_test_mat, display_labels=form_names)
    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
    ax.set_title('Overall testing confusion matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_test_conf_mat.png'))

    # Most commonly selected features
    selected_feats = np.zeros(num_feats)
    for i in range(1, num_subs):
        sub_file = os.path.join(args.save_dir, 'stats', f'subject_{i}.npz')
        sub_results = np.load(sub_file)
        selected_feats[sub_results['selected_inds']] += 1

    filter_plot_info = np.zeros((num_feats, 2))
    filter_plot_info[:, 0] = np.arange(num_feats)
    filter_plot_info[:, 1] = selected_feats
    filter_plot_filename = os.path.join(args.save_dir, 'plots', 'selected_feats.png')
    plot_feat(filter_plot_info, feat_names, args.num_vis_feat, 'Most Frequently Selected Features', 
        'Times Selected', filter_plot_filename)

    # Average filter scores
    avg_filter_scores = np.zeros(num_feats)
    for i in range(1, num_subs):
        sub_file = os.path.join(args.save_dir, 'stats', f'subject_{i}.npz')
        sub_results = np.load(sub_file)
        avg_filter_scores += sub_results['filter_scores']

    avg_filter_scores /= num_subs
    avg_filter_plot_info = np.zeros((num_feats, 2))
    avg_filter_plot_info[:, 0] = np.arange(num_feats)
    avg_filter_plot_info[:, 1] = avg_filter_scores
    avg_filter_plot_filename = os.path.join(args.save_dir, 'plots', 'avg_filter_scores.png')
    plot_feat(avg_filter_plot_info, feat_names, args.num_vis_feat, 'Average Filter Scores',
        'Filter Score', avg_filter_plot_filename)

    return

def main():
    args = arg_parse()

    if args.do_train:
        train(args)

    if args.do_plot:
        visualize(args)

    return

if __name__ == '__main__':
    main()
