'''
Start code for Project 1-Part 1 and optional 2. 
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Prameth Gaddale
    PSU Email ID: pqg5273@psu.edu
    Description: (A short description of what each of the functions you're written does).
}
'''


import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# TODO: write your MaximalLikelihood class 
class ML():
    def __init__(self, degree = None):
        """
        Initializes the Maximum Likelihood model. You may add any parameters/variables you need.
        You may also add any functions you may need to the class.
        """
        self.degree = degree

    def fit(self, x, y):
        """
        Fits the Maximum Likelihood model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))
        intermediate = (x.T)@x
        weights = np.linalg.solve(intermediate, (x.T)@y)
        return weights

    def predict(self, x, weights):
        """
        Predicts the targets of the data.
        Args:
            x (np.array)      : The features of the data.
            weights (np.array): Weights from the ML.fit() method.
        Returns:
            np.array: The predicted targets of the data.
        """
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))
        predictions = np.dot(x, weights)
        return predictions


# TODO: write your MAP class
class MAP():
    def __init__(self, alpha=0.005, beta=11.1, lnlambda = None, customReguralization = False, degree = None):
        """
        Initializes the Maximum A Posteriori model. You may add any parameters/variables you need.
        Args:
            alpha (float): The alpha parameter of the model.
            beta (float): The beta parameter of the model.

        You may also add any functions you may need to the class.
        """
        self.alpha = alpha
        self.beta = beta
        self.lnlambda = lnlambda
        self.customReguralization = customReguralization
        self.degree = degree

    def fit(self, x, y):
        """
        Fits the Maximum A Posteriori model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        """
        if self.customReguralization:
            lnlambda = self.lnlambda
        else:
            lnlambda = np.log(self.alpha/self.beta)

        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))
        intermediate = ((x.T)@x) + np.exp(lnlambda)*np.eye(x.shape[1], x.shape[1])
        weights = np.linalg.solve(intermediate, (x.T)@y)
        return weights

    def predict(self, x, weights):
        """
        Predicts the targets of the data.
        Args:
            x (np.array): The features of the data.
            weights (np.array): The weights from the MAP.fit() method.
        Returns:
            np.array: The predicted targets of the data.
        """
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))
        predictions = np.dot(x, weights)
        return predictions

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