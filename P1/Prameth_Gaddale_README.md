# Project 1
Name:   Prameth Gaddale
Email: pqg5273@psu.edu

## File Structure

![Hierarchy](Hierarchy.png)

### `models.py`
The file contains the classes for the Maximum Likelihood Estimation and Maximum A Posteriri Model.
- `ML()` class contains the functions `fit()` and `predict()`.
- `MAP()` class contains the functions `fit()` and `predict()`.

### `linear_regression.py`
The file contains the following functions:
- `generateNoisyData()`
    - Generates noisy sample points and saves the data. The function will save the data as a npz file.
- `plot_with_shadded_bar()`
    - Plots the GT data for visualization.
- `plot_results()`
    - Plots the GT data and model prediction data for visualization.
- `rmse()`
    - Calculates the root mean squared error betwwen the predicted values and the target values.
- `zero_pad_weights()`
    - Outputs the zero-padded weights for printing various model weights as a Pandas DataFrame.
- `linear_regression()`
    - Main Linear Regression File

### `classification.py`
The file contains the following functions:
- `viz_desc_bounds()`
    - Visualizes the decision boundaries of a classifier trained on two features of the dataset.
- `load_dataset()`
    - Loads the taiji dataset.
- `plot_conf_mats()`
    - Plots the confusion matrices for the training and testing data.
- `plot_class_mats()`
    - Plots the classification rates for each class in the form of a matrix for the training and test sets.
- `fisher_projection()`
    - Takes the training features and training labels and calculates the Fisher projection matrix.
- `classification()`
    - Main classificaition function.