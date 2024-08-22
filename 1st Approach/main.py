# ====================================================================================================================================== #
#                        Implementación de una técnica de aprendizaje máquina sin el uso de un framework                                 #
# ====================================================================================================================================== #
#                                                   Gamaliel Marines Olvera - A01708746                                                  #  
#                                                             09/22/2024                                                                 #   
# ====================================================================================================================================== #




# ====================================================================================================================================== #
# This code uses a linear regression algorithm to predict the value of a dependent variable(Y) 
# based on the value of independent variables(Xi).
#
# The algorithm uses the gradient descent optimization technique to minimize the error between the predicted value and the real value.
#
# The code reads a dataset from a CSV file, extracts the features and the target variable, scales the dataset, applies the
# gradient descent algorithm to find the optimal parameters for the linear regression model, and uses mean square error to find the error
#for each epoch.
#
# The code also calculates the mean square error for each epoch and generates a graph to visualize the error reduction.
#
# The dataset used in this code is the "Metro Interstate Traffic Volume" dataset, which contains information about traffic volume.
# The features include:holiday, temperature, rain, snow, clouds, date and time, and weather-description variables.
# The target variable is the traffic volume.
#
# The code consists of the following steps:
# 1. Process Data Set: Load the dataset from a CSV file, extract the features and target variable, and scale the dataset.
# 2. Computation: Apply the gradient descent algorithm to find the optimal parameters for the linear regression model.
# 3. Graph: Generate a graph to visualize the mean square error reduction for each epoch.
#
# The code uses the following functions:
# - hypothesis_function: Computes the hypothesis function for linear regression.
# - mean_square_error_function: Computes the mean square error between the predicted value and the real value.
# - gradient_descent_function: Applies the gradient descent algorithm to update the parameters.
# - scaling_function: Scales the dataset to improve the convergence of the algorithm.
#
# The code uses the following parameters:
# - alfa: The learning rate for the gradient descent algorithm.
# - parameters: The initial parameters for the linear regression model.
# - epoch: The number of iterations for the gradient descent algorithm.
#
# ====================================================================================================================================== #




# ====================================================================================================================================== #
#                                                         IMPORT LIBRARIES                                                               #
# ====================================================================================================================================== #
# import numpy library to use numpy arrays
import numpy as np
# import csv library to read and write csv files (for the dataset)
import csv
# import matplotlib library to generate graphs
import matplotlib.pyplot as plt
# import train_test_split from sklearn to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split


# Global variable to store the error
__error__ = []


# ====================================================================================================================================== #
#                                                          HYPOTHESIS FUNCTION                                                           #
# ====================================================================================================================================== #
#
# The hypothesis function computes the predicted value based on the linear regression model.
# The function takes two parameters: the parameters (θs) of the linear regression model and the features (X) of a data point.
# It computes the dot product of the parameters and features to get the predicted value.
# The function returns the predicted value.
#
# The hypothesis function is defined as:
# hθ(x) = θ0 + θ1*x1 + θ2*x2 + ... + θn*xn
#
# Where:
# - hθ(x) is the predicted value
# - θ0, θ1, θ2, ..., θn are the parameters of the linear regression model
# - x1, x2, ..., xn are the features of the data point
#
# ====================================================================================================================================== #

def hypothesis_function(parameters, x_features):
    add = 0                                                    	 	
    for i in range(len(parameters)):                              			
        add += parameters[i] * x_features[i]   			
    return add                                                 	 	


# ====================================================================================================================================== #
#                                                      MEAN SQUARE ERROR FUNCTION                                                        #
# ====================================================================================================================================== #
# Mean Square Error Function: MSE = 1/n * Σ(X₁ - Y₂)²

# parameters -> Parameter θ or m.
# x_features -> Feature x.
# y_results -> Expected results.


def mean_square_error_function(parameters, x_features, y_results):
    # Variable to store the summation of differences.
    acumulated_error = 0
    for i in range(len(x_features)):
        # HYPOTHESIS FUNCTION
        y_hypothesis = hypothesis_function(parameters, x_features[i])

        # Prints the computed result and the real result
        print("Computed Y:  %f  Real Y: %f " % (y_hypothesis, y_results[i]))
        
        # MSE per case.
        error = y_hypothesis - y_results[i]
        acumulated_error += error ** 2
    
    mean_square_error = acumulated_error / len(x_features)
    
    # Returns the mean square error value instead of appending it directly to __error__
    return mean_square_error

# ====================================================================================================================================== #
#                                                    R SQUARED FUNCTION                                                                  #
# ====================================================================================================================================== #
# R-squared: R² = 1 - (Σ(Yi - Ypred)² / Σ(Yi - Ymean)²)

def calculate_r_squared(parameters, x_features, y_results):
    y_mean = np.mean(y_results)
    ss_total = sum((y_i - y_mean) ** 2 for y_i in y_results)  # Total sum of squares
    ss_residual = sum((y_i - hypothesis_function(parameters, x_i)) ** 2 for x_i, y_i in zip(x_features, y_results))  # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


# ====================================================================================================================================== #
#                                                       GRADIENT DESCENT FUNCTION                                                        #
# ====================================================================================================================================== #
#
# The gradient descent function applies the gradient descent algorithm to update the parameters of the linear regression model.
# The function takes four parameters: the parameters (θs) of the linear regression model, the features (X) of the dataset,
# the target variable (Y) of the dataset, and the learning rate (alfa).
# It computes the gradient of the mean square error with respect to each parameter and updates the parameters accordingly.
# The function returns the updated parameters.
#
# The gradient descent algorithm is defined as:
# θj = θj - α * (1/m) * Σ(hθ(x) - y) * xj
#
# Where:
# - θj is the j-th parameter of the linear regression model
# - α is the learning rate
# - m is the number of data points
# - hθ(x) is the predicted value
# - y is the real value
# - xj is the j-th feature of the data point
#
# ====================================================================================================================================== #


def gradient_descent_function(parameters, x_features, y_results, alfa):
    m = len(x_features)
    gradient_descent = list(parameters)
    for i in range(len(parameters)):
        temp = 0
        for j in range(m):
            error = hypothesis_function(parameters, x_features[j]) - y_results[j]
            temp += error * x_features[j][i]
        gradient_descent[i] = parameters[i] - alfa * (1/m) * temp
    return gradient_descent


    

# ====================================================================================================================================== #
#                                                      		   SCALING DATA SET                                                          #
# ====================================================================================================================================== #
#
# The scaling function scales the dataset to improve the convergence of the gradient descent algorithm.
# The function takes the features (X) of the dataset and scales each feature to the range [0, 1].
# It computes the minimum and maximum values of each feature and scales the values accordingly.
# The function returns the scaled features.
#
# The scaling function is defined as:
# x' = (x - min(x)) / (max(x) - min(x))
#
# Where:
# - x' is the scaled value
# - x is the original value
# - min(x) is the minimum value of the feature
# - max(x) is the maximum value of the feature
#
# ====================================================================================================================================== #

def scaling_function(x_features):
    x_features = np.asarray(x_features).T.tolist()

    for i in range(1, len(x_features)): 
        for j in range(len(x_features[i])):
            acum=+ x_features[i][j]
        avg = acum/(len(x_features[i]))
        max_val = max(x_features[i])
        for j in range(len(x_features[i])):
            x_features[i][j] = (x_features[i][j] - avg)/(max_val)  #Mean scaling
    return np.asarray(x_features).T.tolist() # Aplica el escalado con la media
"""
def scaling_function(x_features):
    x_features = np.asarray(x_features).T.tolist()
    for i in range(1, len(x_features)):
        min_val = min(x_features[i])
        max_val = max(x_features[i])
        for j in range(len(x_features[i])):
            x_features[i][j] = (x_features[i][j] - min_val) / (max_val - min_val)
    return np.asarray(x_features).T.tolist()
"""


# ====================================================================================================================================== #
#                                                         1. Process Data Set                                                            #
# ====================================================================================================================================== #

# Load data from CSV file
# First pass: Calculate the mean for each column
with open("Metro_Interstate_Traffic_Volume.csv", 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)

    x_features = []
    y_results = []

    for row in reader:
        # Parse row data and append to respective lists
        x_features.append([1.0] + [float(x) for x in row[:-1]])
        y_results.append(float(row[-1]))



# Preprocesamiento de datos
x_train, x_test, y_train, y_test = train_test_split(x_features, y_results, test_size=0.2, random_state=42)
x_train = scaling_function(x_train)
x_test = scaling_function(x_test)


# ====================================================================================================================================== #
#                                                         2. Computation                                                                 #
# ====================================================================================================================================== #

# Initial Parameters
epoch = 200
alfa = 0.01
parameters = [0.0] * len(x_features[0])

# TRAINING
for i in range(epoch):
    parameters = gradient_descent_function(parameters, x_train, y_train, alfa)
    error = mean_square_error_function(parameters, x_train, y_train)
    print('Epoch %d: Error = %.8f' % (i, error))
    __error__.append(error)


# Calculate R-squared value
r_squared_train = calculate_r_squared(parameters, x_train, y_train)
r_squared_test = calculate_r_squared(parameters, x_test, y_test)
print("R-squared (Training Set):", r_squared_train)
print("R-squared (Test Set):", r_squared_test)


# ====================================================================================================================================== #
#                                                         3. Graph                                                                       #
# ====================================================================================================================================== #

# Generate a graph to visualize the error reduction for each epoch
plt.plot(range(epoch), __error__)
plt.xlabel("Epoch")
plt.ylabel("Mean Square Error")
plt.title("Mean Square Error Reduction")
plt.show()
