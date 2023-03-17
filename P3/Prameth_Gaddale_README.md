# Project 3
Name: Prameth Gaddale
Email: pqg5273@psu.edu

## File Structure
```
.
└── Prameth_Gaddale_Project3/
    ├── util.py
    ├── models.py
    ├── classification.py
    ├── baseline/
    │   ├── Taiji/
    │   │   ├── full
    │   │   └── lod4
    │   └── Wallpaper/
    │       ├── test
    │       └── test_challenge
    ├── improved/
    │   ├── Taiji/
    │   │   ├── full
    │   │   └── lod4
    │   └── Wallpaper/
    │       ├── test
    │       └── test_challenge
    ├── data_augmented/
    │   ├── baseline/
    │   │   ├── test
    │   │   └── test_challenge
    │   └── vgg16/
    │       ├── test
    │       └── test_challenge
    ├── Prameth_Gaddale_Report.pdf
    └── Prameth_Gaddale_README.md
```

### `models.py`
The model classes for performing classification on the Taiji and Wallpaper datasets.

This file contains the code for the MLP, MLP2, CNN, and CNN2 models.
 - `MLP`: Baseline MLP Model
 - `MLP2`: Improved MLP Model
 - `CNN`: Baseline CNN Model
 - `CNN2`: Improved CNN Model
 - `VGG16`:
     - VGG16 Model for Data Augmentation Section.
     - The first layer has been modified to take in 1 channel instead of 3.
     - The convolution layer in the first layer has been modified to take in 32 filters instead of 64.
     - Also, the last layer has been modified to output 17 classes instead of 1000.

### `classification.py`
The main file to perform classification.
 - `test`: Test the model.
 - `train`: Train the model and periodically log the loss and accuracy.\
 - `visualize_layer`: Visualize the activations of a layer in the model.
 - `visualize_tsne`: Visualize the predictions using t-SNE.
 - `wallpaper_main`: Main function for the wallpaper classification task.
     - Put the --baseline argument to use the baseline model.
     - Put the --improved argument to use the improved model.
 - `taiji_main`: Main function for the taiji classification task.
     - Put the --baseline argument to use the baseline model.
     - Put the --improved argument to use the improved model.
    

### `util.py`
The file containing all the utilities.
 - `arg_parse`: Parses the arguments.
     - Kindly add the arguement of --baseline or --improved to select the model configuration
 - `get_stats`: Calculates the prediction stats
 - `prep_data`: Preprocess the data and labels by turning them into tensors and normalizing
 - `TaijiData`: Dataset class for Taiji dataset
 - `plot_training_curve`: Plots the training curve
 - `visualize`: Visualize the results