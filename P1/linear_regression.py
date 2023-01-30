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
import pandas as pd
from matplotlib import collections as mc
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
    np.savez(f'data{num_points}.npz', x=x, y=y, t=t, sigma=sigma)

# Feel free to change aspects of this function to suit your needs.
# Such as the title, labels, colors, etc.
def plot_with_shadded_bar(x=None, y=None, sigma=None, num_points = 50):
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
    np.load(f'data{num_points}.npz')
    x = np.load(f'data{num_points}.npz')['x']
    y = np.load(f'data{num_points}.npz')['y']
    t = np.load(f'data{num_points}.npz')['t']
    sigma = np.load(f'data{num_points}.npz')['sigma']

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

def plot_results(x=None, y=None, pred1 = None, label1 = None, sigma = False, pred2 = None, label2 = None, title = None, file_name = None, num_points = 50):
    """
    Plots the GT data for visualization.
    Args:
        x: x values
        y: y values
        pred: prediction values
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    np.load(f'data{num_points}.npz')
    x = np.load(f'data{num_points}.npz')['x']
    y = np.load(f'data{num_points}.npz')['y']
    t = np.load(f'data{num_points}.npz')['t']
    sigma = np.load(f'data{num_points}.npz')['sigma']

    # Noise distribution for plotting the results.
    nmu = 0
    sigma = 0.3
    noise = nmu + sigma * np.random.randn(num_points)

    # Plot the ground truth curve of the generated data.
    fig, ax = plt.subplots()

    ax.plot(x, y, 'g', label = "Ground Truth")
    ax.plot(x, pred1, 'r', label = label1)
    if pred2 is not None:
        ax.plot(x, pred2, "y", label = label2)
        ax.fill_between(x, y-sigma, y+sigma, color='r', alpha=0.2)

    # Plot the noisy data points.
    ax.scatter(x, t, label='Noisy Data Points')
    if pred2 is None:
        ax.fill_between(x, pred1-sigma, pred1+sigma, color='r', alpha=0.2)
        if num_points == 50:
            lines = [[(i, j), (i, line)] for i, j, line in zip(x, pred1, t)]    
            lc = mc.LineCollection(lines, colors='red', linewidths=1, zorder=1)
            ax.add_collection(lc)

    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values, Predictions, Noisy Data')
    ax.set_title(title)
    plt.legend()
    plt.savefig(f'results/{file_name}.png')
    # plt.show()
    plt.close(fig)

def rmse(a, b):
    return np.sqrt(np.mean(np.square(a-b)))

def pad_weights(weights):
    while len(weights) <= 9:
        weights = np.append(weights, 0.)
    return weights

# TODO: Use the existing functions to create/load data as needed. You will now call your linear regression model
# to fit the data and plot the results.
def linear_regression(num_points = 50, calcRMSE = False, rmse_degree = 9):
    np.load(f'data{num_points}.npz')
    x = np.load(f'data{num_points}.npz')['x']
    y = np.load(f'data{num_points}.npz')['y']
    t = np.load(f'data{num_points}.npz')['t']
    sigma = np.load(f'data{num_points}.npz')['sigma']

    ml_weights_dict, map_weights_dict = {}, {}

    for i in [0,1,3,6,9]:
        ML_Model = ML(degree = i)
        ml_weights = ML_Model.fit(x, t)
        ml_predictions = ML_Model.predict(x, ml_weights)
        ml_weights_dict[f"ML:{i}"] = pad_weights(ml_weights)
        plot_results(x, y, ml_predictions, title = f"ML Model Degree {i}", file_name = f"{num_points}_ml_{i}", label1 = "ML Model Prediction", num_points = num_points)
    print(f"Number of Points: {num_points}")
    print(pd.DataFrame(ml_weights_dict))

    for i in [0,1,3,6,9]:
        MAP_Model = MAP(degree = i, customReguralization = False)
        map_weights = MAP_Model.fit(x, t)
        map_predictions = MAP_Model.predict(x, map_weights)
        map_weights_dict[f"MAP:{i}"] = pad_weights(map_weights)
        plot_results(x, y, map_predictions, title = f"MAP Model Degree {i}", file_name = f"{num_points}_map_{i}", label1 = "MAP Model Prediction", num_points = num_points)
    print(f"Number of Points: {num_points}")    
    print(pd.DataFrame(map_weights_dict))

    for i in [0,1,3,9,15,20]:
        ML_Model = ML(degree = i)
        MAP_Model = MAP(degree = i, customReguralization = False)
        ml_weights = ML_Model.fit(x, t)
        map_weights = MAP_Model.fit(x, t)
        ml_predictions = ML_Model.predict(x, ml_weights)
        map_predictions = MAP_Model.predict(x, map_weights)
        plot_results(x, y, pred1 = ml_predictions, label1 = f"ML Model Degree: {i}", \
                        title = f"ML vs MAP Predictions Degree: {i}", \
                        pred2 = map_predictions, label2 = f"MAP Model Degree: {i}", \
                        file_name = f"{num_points}_ml_vs_map{i}", num_points = num_points)
    if calcRMSE:
        rmse_arr = []
        for lnlambda in range(-40, 10):
            inputs = np.load("data50.npz")["x"]
            targets = np.load("data50.npz")["t"]
            rmseModel = MAP(degree = rmse_degree, customReguralization = True, lnlambda = lnlambda)
            weights = rmseModel.fit(inputs, targets)
            predictions = rmseModel.predict(inputs, weights)
            rmse_error = rmse(predictions, targets)
            rmse_arr.append(rmse_error)
        plt.plot(range(-40,10), rmse_arr)
        # plt.ylim([0, 1])
        plt.xlabel(r"ln$\lambda$")
        plt.ylabel(r"$E_{RMS}$")
        plt.legend()
        plt.savefig("results/rmse-lnlambda.png")
        # plt.show()

    for lnlambda in [-18, -15, -13, 0, 10, 20]:
        CustomModel = MAP(degree = 3, customReguralization = True, lnlambda = lnlambda)
        custom_weights = CustomModel.fit(x, t)
        custom_predictions = CustomModel.predict(x, custom_weights)
        plot_results(x, y, custom_predictions, title = r"Custom Model Degree 3, ln$\lambda$ = " + str(lnlambda), file_name = f"{num_points}_lnlambda{lnlambda}", label1 = r"$ln\lambda$ = "+str(lnlambda)+" Model", num_points = num_points)

def main():
    for num_points in [50, 20, 200]:
        generateNoisyData(num_points = num_points)
        linear_regression(num_points = num_points, calcRMSE = True)
        plot_with_shadded_bar()

if __name__ == '__main__':
    main()