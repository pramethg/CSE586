'''
Start code for Project 1-Part 1: Linear Regression. 
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
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO: import your models
from models import ML, MAP

def generateNoisyData(num_points=50):
    """
    Generates noisy sample points and saves the data. The function will save the data as a npz file.
    Args:
        num_points: number of sample points to generate.
    """
    x = np.linspace(1, 4*math.pi, num_points)
    y = np.sin(x*0.5)

    # Define the noise model
    nmu = 0
    sigma = 0.3
    noise = nmu + sigma * np.random.randn(num_points)
    t = y + noise

    # Save the data
    np.savez('data.npz', x=x, y=y, t=t, sigma=sigma)

# Feel free to change aspects of this function to suit your needs.
# Such as the title, labels, colors, etc.
def plot_with_shadded_bar(x=None, y=None, sigma=None):
    """
    Plots the GT data for visualization.
    Args:
        x: x values
        y: y values
        sigma: standard deviation
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    # Example plotting with the GT data, you can use this as a reference. You will later 
    # use this function to plot the results of your model.
    np.load('data.npz')
    x = np.load('data.npz')['x']
    y = np.load('data.npz')['y']
    t = np.load('data.npz')['t']
    sigma = np.load('data.npz')['sigma']

    # Plot the ground truth curve of the generated data.
    fig, ax = plt.subplots()

    # Plot curve with red shaded region spans on std.
    ax.plot(x, y, 'r', label='Ground Truth Example')
    ax.fill_between(x, y-sigma, y+sigma, color='r', alpha=0.2)

    # Plot the noisy data points.
    ax.scatter(x, t, label='Noisy Data')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Data Point Plot Example')

    plt.savefig('results/gt_data.png')
    plt.close(fig)

def plot_results(x=None, y=None, pred1 = None, label1 = None, pred2 = None, label2 = None, title = None, file_name = None):
    """
    Plots the GT data for visualization.
    Args:
        x: x values
        y: y values
        pred: prediction values
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    np.load('data.npz')
    x = np.load('data.npz')['x']
    y = np.load('data.npz')['y']
    t = np.load('data.npz')['t']

    # Plot the ground truth curve of the generated data.
    fig, ax = plt.subplots()

    ax.plot(x, y, 'g', label = "Ground Truth")
    ax.plot(x, pred1, 'r', label = label1)
    if pred2 is not None:
        ax.plot(x, pred2, "y", label = label2)

    # Plot the noisy data points.
    ax.scatter(x, t, label='Noisy Data')

    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values, Predictions, Noisy Data')
    ax.set_title(title)
    plt.legend()
    plt.savefig(f'results/{file_name}.png')
    plt.show()
    plt.close(fig)


# TODO: Use the existing functions to create/load data as needed. You will now call your linear regression model
# to fit the data and plot the results.
def linear_regression():
    # Load the data
    np.load('data.npz')
    x = np.load('data.npz')['x']
    y = np.load('data.npz')['y']
    t = np.load('data.npz')['t']
    sigma = np.load('data.npz')['sigma']

    ML_Model = ML(degree = 3)
    ml_weights = ML_Model.fit(x, t)
    ml_predictions = ML_Model.predict(x, ml_weights)
    plot_results(x, y, ml_predictions, title = "ML Model Degree 3", file_name = "ml_3", label1 = "ML Model Predictions")

    MAP_Model = MAP(degree = 3, customReguralization = False)
    map_weights = MAP_Model.fit(x, t)
    map_predictions = MAP_Model.predict(x, map_weights)
    plot_results(x, y, map_predictions, title = "MAP Model Degree 3", file_name = "map_3", label1 = "MAP Model Predictions")

    plot_results(x, y, pred1 = ml_predictions, label1 = "ML Model Degree 3", pred2 = map_predictions, label2 = "MAP Model Degree 3", title = "ML vs MAP Predictions", file_name = "mlevsmap")

    lnlambda = -18
    CustomModel = MAP(degree = 3, customReguralization = True, lnlambda = lnlambda)
    custom_weights = CustomModel.fit(x, t)
    custom_predictions = CustomModel.predict(x, custom_weights)
    plot_results(x, y, custom_predictions, title = r"Custom Model Degree 3, ln$\lambda$ = " + str(lnlambda), file_name = "lnlambda-18", label1 = r"$ln\lambda$ = "+str(lnlambda)+" Model")

def main():
    generateNoisyData(50)
    plot_with_shadded_bar()
    linear_regression()


if __name__ == '__main__':
    main()