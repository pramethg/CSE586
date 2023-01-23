'''
Start code for Project 1-Part 1 and optional 2. 
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name:
    PSU Email ID:
    Description: (A short description of what each of the functions you're written does).
}
'''


import math
import numpy as np

# TODO: write your MaximalLikelihood class 
class ML():
    def __init__(self):
        """
        Initializes the Maximum Likelihood model. You may add any parameters/variables you need.
        You may also add any functions you may need to the class.
        """
        raise NotImplementedError
  

    def fit(self, x, y):
        """
        Fits the Maximum Likelihood model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Predicts the targets of the data.
        Args:
            x (np.array): The features of the data.
        Returns:
            np.array: The predicted targets of the data.
        """
        raise NotImplementedError


# TODO: write your MAP class
class MAP():
    def __init__(self, alpha=0.005, beta=11.1):
        """
        Initializes the Maximum A Posteriori model. You may add any parameters/variables you need.
        Args:
            alpha (float): The alpha parameter of the model.
            beta (float): The beta parameter of the model.

        You may also add any functions you may need to the class.
        """
        self.alpha = alpha
        self.beta = beta

    def fit(self, x, y):
        """
        Fits the Maximum A Posteriori model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Predicts the targets of the data.
        Args:
            x (np.array): The features of the data.
        Returns:
            np.array: The predicted targets of the data.
        """
        raise NotImplementedError

# Optional: If you choose to implement a classifier, please do so in this class
class Classifier():
    def __init__(self, params=None):
        """
        Initializes the classifier. You may add any parameters you want. 

        You may also add any functions you may need to the class.
        """
        self.params = params

    def fit(self, x, y):
        """
        Fits the classifier to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Predicts on the data x.
        Args:
            x (np.array): The features of the data.

        Returns:
            out (np.array): target predictions.
        """
        out = x
        raise NotImplemented