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
    Description: 
                generateNoisyData(): Takes in the number of points as input and saves the generated points as a .npz file.
                    CHANGES: Changed the format of the datafile name that takes in the number of points as a parameter while saving the file
                    e.g.: data50.npz: 50 data points, data200.npz: 200 datapoints.

                plot_with_shadded_bar(): Plots the results and saves in .png format.

                plot_results(): Plots the results of the model predictions.
                    Parameters:
                    x (np.array): Input Data,
                    y (np.array): Ground Truth Data,
                    pred1 (np.array): 1st model prediction 
                    label1 (str): Label of 1st model prediction 
                    pred1 (np.array): 2nd model prediction 
                    label2 (str): Label of 2nd model prediction
                    title (str): Title of the plot
                    file_name (str): Filename for the saved plot
                    num_points (int): Number of points used for the experiment
                
                rmse(): Calculates the root-mean-squared error.

                zero_pad_weights(): For printing the comparision table for various model degrees through a Pandas DataFrame.
                This function just converts all the model degree weights to match equal dimensions with zeroes.

                linear_regression(): Main linear regression function corresponding to all the experiments.             

}
'''

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections as mc
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

def plot_results(x=None, y=None, pred1 = None, label1 = None, pred2 = None, label2 = None, title = None, file_name = None, num_points = 50):
    """
    Plots the GT data for visualization.
    Args:
        x (np.array): Input Data,
        y (np.array): Ground Truth Data,
        pred1 (np.array): 1st model prediction 
        label1 (str): Label of 1st model prediction 
        pred1 (np.array): 2nd model prediction 
        label2 (str): Label of 2nd model prediction
        title (str): Title of the plot
        file_name (str): Filename for the saved plot
        num_points (int): Number of points used for the experiment
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    # Loads the dataset from the saved data files. The file has been changes to suffice changes in number of points.
    np.load(f'data{num_points}.npz')
    x = np.load(f'data{num_points}.npz')['x']
    y = np.load(f'data{num_points}.npz')['y']
    t = np.load(f'data{num_points}.npz')['t']
    sigma = np.load(f'data{num_points}.npz')['sigma']

    # Used for plotting the results.
    sigma = 0.3

    # Plot the ground truth curve of the generated data.
    fig, ax = plt.subplots()

    ax.plot(x, y, 'g', label = "Ground Truth")

    # Plot Model 1 Predictions
    ax.plot(x, pred1, 'r', label = label1)
    # If Model 2 is used for comparision, plot the Model 2 predictions
    if pred2 is not None:
        ax.plot(x, pred2, "y", label = label2)
        # Fill in the region of the std. ground truth with sigma
        ax.fill_between(x, y-sigma, y+sigma, color='r', alpha=0.2)

    # Plot the noisy data points.
    ax.scatter(x, t, label='Noisy Data Points')

    # If only 1 Model prediction is being plotted
    if pred2 is None:
        # Fill the region spanned by the std. sigma model 1 prediction with red.
        ax.fill_between(x, pred1-sigma, pred1+sigma, color='r', alpha=0.2)
        # Only when the number of points is 50, draw lines from the noisy datapoints to the model curve.
        if num_points == 50:
            lines = [[(i, j), (i, line)] for i, j, line in zip(x, pred1, t)]    
            lc = mc.LineCollection(lines, colors='red', linewidths=1, zorder=1)
            ax.add_collection(lc)

    # Create labels, title, legend and file that takes in the file_name parameter.
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values, Predictions, Noisy Data')
    ax.set_title(title)
    plt.legend()
    plt.savefig(f'results/{file_name}.png')
    # plt.show()
    plt.close(fig)

def rmse(predictions, targets):
    """
    Calculates the root mean squared error betwwen the predicted values and the target values.
    Args:
        predictions (np.array)
        targets (np.array)
    """
    return np.sqrt(np.mean(np.square(predictions-targets)))

def zero_pad_weights(weights):
    """
    Outputs the zero-padded weights for printing various model weights as a Pandas DataFrame.
    Args:
        predictions (np.array)
        targets (np.array)
    """
    while len(weights) <= 9:
        weights = np.append(weights, 0.)
    return weights

def linear_regression(num_points = 50, calc_rmse = False, rmse_degree = 9):
    '''
    Main Linear Regression Function
    Args:
            num_points (int): Number of points for the given experiment.
            calc_rmse (bool): If true, will generate the plot of RMSE Error vs ln(lambda)
            rmse_degree (int): Represents the degree of the polynomial model that is used for the RMSE plot.
    '''

    # Load the dataset corresponding to the given number of points.
    np.load(f'data{num_points}.npz')
    x = np.load(f'data{num_points}.npz')['x']
    y = np.load(f'data{num_points}.npz')['y']
    t = np.load(f'data{num_points}.npz')['t']
    sigma = np.load(f'data{num_points}.npz')['sigma']

    # Dictionaries for saving the weights of the ML and MAP models.
    ml_weights_dict, map_weights_dict = {}, {}

    # For Maximum Likelihood model creates polynomial model with various degrees and prints/saves the predictions.
    # Loop over various model degrees.
    for i in [0,1,3,6,9]:

        # Create a ML model object.
        ML_Model = ML(degree = i)

        # Calculates the weights from the ML.fit() class methods.
        ml_weights = ML_Model.fit(x, t)

        # Calculates the model predictions from the ML.predict() class method.
        ml_predictions = ML_Model.predict(x, ml_weights)

        # Saves the model weights in the dictionary with the degree as the key.
        ml_weights_dict[f"ML:{i}"] = zero_pad_weights(ml_weights)

        # Plots and saves the results.
        plot_results(x, y, ml_predictions, title = f"ML Model Degree {i}", file_name = f"{num_points}_ml_{i}", label1 = "ML Model Prediction", num_points = num_points)
    print(f"Number of Points: {num_points}")
    print(pd.DataFrame(ml_weights_dict))

    # For Maximum A Posteriori model creates polynomial model with various degrees and prints/saves the predictions.
    # Loop over various model degrees.
    for i in [0,1,3,6,9]:

        # Create a ML model object.
        MAP_Model = MAP(degree = i, customReguralization = False)

        # Calculates the weights from the ML.fit() class methods.
        map_weights = MAP_Model.fit(x, t)

        # Calculates the model predictions from the ML.predict() class method.
        map_predictions = MAP_Model.predict(x, map_weights)

        # Saves the model weights in the dictionary with the degree as the key.
        map_weights_dict[f"MAP:{i}"] = zero_pad_weights(map_weights)

        # Plots and saves the results.
        plot_results(x, y, map_predictions, title = f"MAP Model Degree {i}", file_name = f"{num_points}_map_{i}", label1 = "MAP Model Prediction", num_points = num_points)
    print(f"Number of Points: {num_points}")    
    print(pd.DataFrame(map_weights_dict))

    # Iterates over various model degrees and compares the predictions of ML and MAP models.
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

    # If calc_rmse is True, lnlambda iterates over range(-40,10) and calculates the root mean sqaured error.
    # Results are plotted and saved.
    if calc_rmse:
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

    # Iterate over various lnlambda values and plot the model predictions.
    for lnlambda in [-18, -15, -13, 0, 10, 20]:
        CustomModel = MAP(degree = 3, customReguralization = True, lnlambda = lnlambda)
        custom_weights = CustomModel.fit(x, t)
        custom_predictions = CustomModel.predict(x, custom_weights)
        plot_results(x, y, custom_predictions, title = r"Custom Model Degree 3, ln$\lambda$ = " + str(lnlambda), file_name = f"{num_points}_lnlambda{lnlambda}", label1 = r"$ln\lambda$ = "+str(lnlambda)+" Model", num_points = num_points)

def main():
    # Iterate over various number of points to perform linear regression experiments.
    for num_points in [50, 20, 200]:
        generateNoisyData(num_points = num_points)
        linear_regression(num_points = num_points, calc_rmse = True)
        plot_with_shadded_bar()

if __name__ == '__main__':
    main()