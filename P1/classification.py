'''
Start code for Project 1-Part 2: Classification
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Prameth Gaddale
    PSU Email ID: pqg5273@psu.edu
    Description:
                viz_desc_bounds() : Visualizes the decision boundaries of a classifier trained on two features of the dataset.

                load_dataset() : Loads the Taiji Dataset

                plot_conf_mats() : Plots the confusion matrix of the training and test classifications.

                plot_class_mats() :  Plots the classification rates for each class in the form of a matrix for the training and test sets.

                fisher_projection() : Takes the training features and training labels and calculates the significant Fisher projected eigen vectors.

                classification() : Main classification function with KNN and LDA classifiers.

                EXAMPLE FUNCTIONS:
                    example_decision_boundary() : An example of how to visualize the decision boundary of a classifier.
                    example_classification() : An example of performing the classification.
}
'''

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def viz_desc_bounds(classifier, feats, labels, idxA, idxB):
    """
    Visualizes the decision boundaries of a classifier trained on two features of the dataset.
    Args:
        classifier: linear classifier trained on 2 features.
        feats: features to be used for visualization.
        labels: labels to be used for visualization.
        idxA & idxB: indices of the features to be used for visualization. 
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    ys = np.sort(np.unique(labels))
    y_ind = np.searchsorted(ys, labels)

    fig, ax = plt.subplots()

    x0, x1 = feats[:, 0], feats[:, 1]
    all_feats = np.concatenate((x0, x1))
    pad = np.percentile(all_feats, 60)

    x_min, x_max = x0.min() - pad, x0.max() + pad
    y_min, y_max = x1.min() - pad, x1.max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    preds = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    preds = preds.reshape(xx.shape)

    lut = np.sort(np.unique(labels)) 
    ind = np.searchsorted(lut,preds)

    markers = ["o", "v", "P", "X", "s", "p", "h", "d"]
    ax.contourf(xx, yy, preds, cmap=plt.cm.Pastel1, alpha=0.8)
    for i in range(len(lut)):
        ax.scatter(x0[y_ind == i], x1[y_ind == i], color=plt.cm.jet(i/len(lut)), s=50, edgecolors='k', marker=markers[i])

    ax.set_xlabel(f'Feature {idxA}')
    ax.set_ylabel(f'Feature {idxB}')
    ax.set_title('Decision Boundary')

    handles = []
    markers = ["o", "v", "P", "X", "s", "p", "h", "d"]
    handles = [plt.plot([],[],color=plt.cm.jet(i/len(lut)), ls="", marker=markers[i])[0] for i in range(len(lut))]
    labels = [f'Class {i}' for i in lut]
    ax.legend(handles, labels, loc='upper right')
    plt.savefig('results/decision_boundary.png')
    plt.show()


def load_dataset(dataset='taiji', verbose=False, subject_index=3):
    '''
    Loads the taiji dataset.
    Args:
        dataset: name of the dataset to load. Currently only taiji is supported.
        verbose: print dataset information if True.
        subject_index: subject index to use for LOSO. The subject with this index will be used for testing. 

    Returns (all numpy arrays):
        train_feats: training features.
        train_labels: training labels.
        test_feats: testing features.
        test_labels: testing labels.
    '''

    if dataset == 'taiji':
        labels = np.loadtxt("data/taiji/taiji_labels.csv", delimiter=",", dtype=int)
        person_idxs = np.loadtxt("data/taiji/taiji_person_idx.csv", delimiter=",", dtype=int)
        feats = np.loadtxt("data/taiji/taiji_quat.csv", delimiter=",", dtype=float)

        # Combine repeated positions
        labels[labels == 4] = 2
        labels[labels == 8] = 6

        # Remove static dimensions. Get mask of all features with zero variance
        feature_mask = np.var(feats, axis=1) > 0

        # Train mask
        train_mask = person_idxs != subject_index

        train_feats = feats[feature_mask, :][:, train_mask].T
        train_labels = labels[train_mask].astype(int)
        test_feats = feats[feature_mask, :][:, ~train_mask].T
        test_labels = labels[~train_mask].astype(int)


    if verbose:
        print(f'{dataset} Dataset Loaded')
        print(f'\t# of Classes: {len(np.unique(train_labels))}')
        print(f'\t# of Features: {train_feats.shape[1]}')
        print(f'\t# of Training Samples: {train_feats.shape[0]}')
        print('\t# per Class in Train Dataset:')
        for cls in np.unique(train_labels):
            print (f'\t\tClass {cls}: {np.sum(train_labels == cls)}')
        print(f'\t# of Testing Samples: {test_feats.shape[0]}')
        print('\t# per Class in Test Dataset:')
        for clas in np.unique(test_labels):
            print(f'\t\tClass {clas}: {np.sum(test_labels == clas)}')
        
    return train_feats, train_labels, test_feats, test_labels

def plot_conf_mats(dataset, **kwargs):
    """
    Plots the confusion matrices for the training and testing data.
    Args:
        dataset: name of the dataset.
        train_labels: training labels.
        pred_train_labels: predicted training labels.
        test_labels: testing labels.
        pred_test_labels: predicted testing labels.
    """

    train_labels = kwargs['train_labels']
    pred_train_labels = kwargs['pred_train_labels']
    test_labels = kwargs['test_labels']
    pred_test_labels = kwargs['pred_test_labels']

    train_confusion = confusion_matrix(train_labels, pred_train_labels)
    test_confusion = confusion_matrix(test_labels, pred_test_labels)

    # Plot the confusion matrices as seperate figures
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion, display_labels=np.unique(train_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title('Training Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_train_confusion.png', bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=np.unique(test_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title('Testing Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_test_confusion.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_class_mats(dataset, **kwargs):
    '''
    Plots the classification rates for each class in the form of a matrix for the training and test sets.
        Args:
        dataset: name of the dataset.
        train_labels: training labels.
        pred_train_labels: predicted training labels.
        test_labels: testing labels.
        pred_test_labels: predicted testing labels.
    '''
    train_labels = kwargs['train_labels']
    pred_train_labels = kwargs['pred_train_labels']
    test_labels = kwargs['test_labels']
    pred_test_labels = kwargs['pred_test_labels']
    
    # Convert the values of the numpy array to floating values
    train_confusion = np.array(confusion_matrix(train_labels, pred_train_labels), dtype = np.float16)

    # Divides the each row with the number of examples in the corresponding to recieve the per-class classification rates.
    for i in range(len(train_confusion)):
        train_confusion[i] = train_confusion[i]/sum(train_confusion[i])
        
    # Rounds the values of the matrix to 3 values after the decimal point. 
    train_confusion = np.around(train_confusion, 3)

    # Convert the values of the numpy array to floating values
    test_confusion = np.array(confusion_matrix(test_labels, pred_test_labels), dtype = np.float16)

    # Divides the each row with the number of examples in the corresponding to recieve the per-class classification rates.
    for i in range(len(test_confusion)):
        test_confusion[i] = test_confusion[i]/sum(test_confusion[i])

    # Rounds the values of the matrix to 3 values after the decimal point.
    test_confusion = np.around(test_confusion, 3)

    # Plots the classification rate matrices.
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion, display_labels=np.unique(train_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title('Training Classification Rate Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_train_classification.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=np.unique(test_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title('Testing Classification Rate Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_test_classification.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def example_decision_boundary(dataset='taiji', indices=[0, 6]):
    """
    An example of how to visualize the decision boundary of a classifier.
    """
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)

    dc_train_feats = train_feats[:, indices]
    dc_test_feats = test_feats[:, indices]

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(dc_train_feats, train_labels)

    # Visualize the decision boundary
    viz_desc_bounds(clf, dc_test_feats, test_labels, indices[0], indices[1])

def example_classification(dataset='taiji'):
    """
    An example of performing classification. Except you will need to first project the data.
    """
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_feats, train_labels)

    # Predict the labels of the training and testing data
    pred_train_labels = clf.predict(train_feats)
    pred_test_labels = clf.predict(test_feats)

    # Get statistics
    plot_conf_mats(dataset, train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels, pred_test_labels=pred_test_labels)

def fisher_projection(train_feats, train_labels):
    '''
    Takes the training features and training labels and calculates the Fisher projection vectors.
    Args:
        train_feats (np.array): Training set features
        train_labels (np.array): Training set labels
    '''
    # Create a list for storing means
    means = []

    # Find the unique training labels and store them.
    train_labels_unique = np.unique(train_labels)

    # Iterate over all the unique training labels and append means to the main means list.
    for label in train_labels_unique:
        matched_train_feats = train_feats[train_labels == label]
        mean_int_train_feats = np.mean(matched_train_feats, axis=0)
        means.append(mean_int_train_feats)

    # Store the global mean of the training features.
    global_mean = np.mean(train_feats, axis=0)

    # Calculate the S_B matrix as shown in the derivations.
    # Initializes with zeroes and then iterates over the means to calculate S_B Matrix
    S_B_Matrix = np.zeros((train_feats.shape[1], train_feats.shape[1])) # (64,64) matrix
    for label, mean in enumerate(means):
        matched_feats = train_feats[train_labels == label].shape[0]
        mean_label = mean - global_mean
        mean_label = mean_label.reshape(-1, 1)
        S_B_Matrix += matched_feats * np.dot(mean_label, mean_label.T)

    # Initialize the S_W matrix to zeroes and then iterate over the means.
    # Uses the closed form implementation shown in the derivations.
    S_W_Matrix = np.zeros((train_feats.shape[1], train_feats.shape[1])) # (64,64) matrix
    S_W_Matrix_Class = []
    for label, mean in enumerate(means):
        S = np.zeros((train_feats.shape[1], train_feats.shape[1])) # (64,64) matrix
        for match in train_feats[train_labels == label]:
            difference = (match - mean)
            difference = difference.reshape(-1, 1)
            S += difference @ difference.T
        S_W_Matrix_Class.append(S)

    # Add each value of the class based s_w matrix to the final S_W matrix.
    for s_w in S_W_Matrix_Class:
        S_W_Matrix += s_w

    # Find the eigen values and eigen vectors of the matrix (S_W)^-1*S_B.
    W = np.linalg.inv(S_W_Matrix) @ S_B_Matrix
    eigen_values, eigen_vectors = np.linalg.eig(W)

    # Store the sorted eigen values as eigen_indices to form the significance order of eigen values.
    eigen_indices = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, eigen_indices]

    # Picks the 8 most significant eigen vectors and outputs them for classification.
    eigen_vectors = eigen_vectors[:, :8] #(64,8) matrix
    return eigen_vectors

def classification(dataset = "taiji", classifier = "lda"):
    '''
    Performs Classification on the Taiji dataset
    '''
    # Loads the dataset
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset, verbose = True)

    # Loads the eigen vectors corresponding to 8 most significant eigen values from the fisher projection function
    eigen_vectors = fisher_projection(train_feats, train_labels)

    # Calculates the fisher projected training features
    proj_train_feats = np.real(train_feats @ eigen_vectors)

    # Calculates the fisher projected test set features
    proj_test_feats = np.real(test_feats @ eigen_vectors)

    # Performs the Classification either through LDA() or KNN() model.
    if classifier == "lda":
        clf = LinearDiscriminantAnalysis()
    if classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors = 10)

    # Classifier object gets fit from the fisher projected training set and training set labels.
    clf.fit(proj_train_feats, train_labels)

    # Predicted training and test set labels are calculated.
    pred_train_labels = clf.predict(proj_train_feats)
    pred_test_labels = clf.predict(proj_test_feats)

    # Using sklearn.metrics.accuracy_score, the overall classification rate has been calculated.
    # The overall classification rate is limited to 5 digits after the decimal point.
    print("Overall Train Classification Rate: ", np.round(accuracy_score(train_labels, pred_train_labels), 5))
    print("Overall Test Classification Rate: ", np.round(accuracy_score(test_labels, pred_test_labels), 5))

    # Plots the confusion matrices.
    plot_conf_mats(dataset, train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels, pred_test_labels=pred_test_labels)
    # Plots the classification rate matrices.
    plot_class_mats(dataset, train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels, pred_test_labels=pred_test_labels)

def main():
    example_decision_boundary()
    # Classifier takes in arguments:
    # classifier: "lda" or "knn".
    classification(classifier = "lda")
    
if __name__ == '__main__':
    main()
