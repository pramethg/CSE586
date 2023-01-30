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
    Description:The classes ML() and MAP() represent the model classes of the Maximum Likelihood,
                and Maximum-A-Posteriori Models.

                ML() class contains various class methods and variables:
                    self.degree: represents the degree of the polynomial model.
                    fit(): Takes the input features, x and targets, y and outputs the corresponding optimal weights when the closed form equation of Maximum Likelihood is applied.
                    predict(): Takes the input features(in terms of the design matrix) and the weights provided from the fit() to give the ML model predictions.

                MAP() class contains various class methods and variables:
                    self.degree: represents the degree of the polynomial model.
                    self.alpha and self.beta values are corresponding the model distribution.
                    self.customReguralization: boolean value that changes the lambda value while performing linear regression for custom lambda values.
                    self.lnlambda: integer that represents the ln(lambda) value for custom reguralization.
                    fit(): Takes the input features, x and targets, y and outputs the corresponding optimal weights when the closed form equation of Maximum A Posteriori is applied.
                    predict(): Takes the input features(in terms of the design matrix) and the weights provided from the fit() to give the MAP model predictions.
}
'''


import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class ML():
    def __init__(self, degree = None):
        """
        Initializes the Maximum Likelihood model.
            self.degree: represents the degree of the polynomial model.
        """
        self.degree = degree

    def fit(self, x, y):
        """
        Fits the Maximum Likelihood model to the data.
        Args:
            x (np.array): The features of the data.
            y (np.array): The targets of the data.
        Returns:
            weights (np.array): The weight/parameters of the ML polynomial model fit.
        """
        # Construct a design matrix that could accomodate higher order polynomial fits
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))
        # Performs the dot product of (X.transpose) and X
        intermediate = (x.T)@x
        # Calculates the weight matrix by solving the closed form equation from the Maximum Likelihood
        # W = ((X^T*X)^-1)X^T*y
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
        # Constructs the design matrix that could accomodate higher order polynomial fits
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))

        # Takes the dot product between input design matrix and the weight vector to return the model predictions.
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
            self.degree: represents the degree of the polynomial model.
            self.alpha and self.beta values are corresponding the model distribution.
            self.customReguralization: boolean value that changes the lambda value while performing linear regression for custom lambda values.
            self.lnlambda: integer that represents the ln(lambda) value for custom reguralization.
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
        Returns:
            weights (np.array): The weights/parameters from the MAP polynomial model fit
        """
        # If customReguralization is True, user can pass custom value of lnlambda.
        if self.customReguralization:
            self.lnlambda = self.lnlambda
        # If customReguralization is False, lnlambda = log(alpha/beta) is calculated.
        else:
            self.lnlambda = np.log(self.alpha/self.beta)

        # Constructs the design matrix that could accomodate higher order polynomial fits
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))

        # Intermediate step in the calculation of the MAP model weights.
        # np.exp is included to pass the value of lambda as, exp(log(lambda)) = lambda
        intermediate = (x.T)@x + np.exp(self.lnlambda)*np.eye(x.shape[1], x.shape[1])

        # Calculates and returns the MAP model weights.
        # W = ((X^T*X)^-1 + lambda*I)X^T*y
        weights = np.linalg.solve(intermediate, (x.T)@y)

        return weights

    def predict(self, x, weights):
        """
        Predicts the targets of the data.
        Args:
            x (np.array): The features of the data.
            weights (np.array): The weights from the MAP.fit() method.
        Returns:
            predictions (np.array): The predicted targets of the data.
        """
        # Constructs the design matrix that could accomodate higher order polynomial fits
        x = PolynomialFeatures(degree = self.degree).fit_transform(x.reshape(-1,1))

        # Takes the dot product between input design matrix and the weight vector to return the model predictions.
        predictions = np.dot(x, weights)

        return predictions
