# ====================================================================================================================================== #
#                        Implementación de una técnica de aprendizaje máquina sin el uso de un framework                                 #
# ====================================================================================================================================== #
#                                                   Gamaliel Marines Olvera - A01708746                                                  #  
#                                                             09/22/2024                                                                 #   
#                                                                                                                                        #        
# ====================================================================================================================================== #

# ====================================================================================================================================== #
# This code uses a linear regression algorithm to predict the value of a dependent variable (Y) based on the value of independent 
# variables (Xi).
#
# The algorithm uses the gradient descent optimization technique to minimize the error between the predicted value and the real value.
#
# The code reads a dataset from a CSV file, extracts the features and the target variable, scales the dataset, applies the
# gradient descent algorithm to find the optimal parameters for the linear regression model, and uses mean square error to find the error
# for each epoch.
#
# The code also calculates the mean square error for each epoch and generates a graph to visualize the error reduction.
#
# Additionally, the R-squared metric is calculated to evaluate the performance of the model.
#
# The dataset used in this code is the "Metro Interstate Traffic Volume" dataset, which contains information about traffic volume.
# The features include: holiday, temperature, rain, snow, clouds, date and time, and weather-description variables.
# The target variable is the traffic volume.
#
# The code consists of the following steps:
# 1. Process Data Set: Load the dataset from a CSV file, extract the features and target variable, and scale the dataset.
# 2. Computation: Apply the gradient descent algorithm to find the optimal parameters for the linear regression model.
# 3. Graph: Generate a graph to visualize the mean square error reduction for each epoch.
# 4. Print Parameters: Display the final parameters after training.
# 5. Prediction: Make a prediction using the trained model and calculate the R-squared value to evaluate the model's performance.
#
# The code uses the following functions:
# - hypothesis_function: Computes the hypothesis function for linear regression.
# - mean_square_error_function: Computes the mean square error between the predicted value and the real value.
# - gradient_descent_function: Applies the gradient descent algorithm to update the parameters.
# - scaling_function: Scales the dataset to improve the convergence of the algorithm.
#
# ====================================================================================================================================== #

# ====================================================================================================================================== #
#                                                         IMPORT LIBRARIES                                                               #
# ====================================================================================================================================== #
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Global variable to store the error
__error__ = []

# ====================================================================================================================================== #
#                                                          HYPOTHESIS FUNCTION                                                           #
# ====================================================================================================================================== #

def hypothesis_function(parameters, x_features):
    return sum(param * feature for param, feature in zip(parameters, x_features))

# ====================================================================================================================================== #
#                                                      MEAN SQUARE ERROR FUNCTION                                                        #
# ====================================================================================================================================== #

def mean_square_error_function(parameters, x_features, y_results):
    acumulated_error = 0
    for i in range(len(x_features)):
        y_hypothesis = hypothesis_function(parameters, x_features[i])
        error = y_hypothesis - y_results[i]
        acumulated_error += error ** 2
    return acumulated_error / len(x_features)

# ====================================================================================================================================== #
#                                                      GRADIENT DESCENT FUNCTION                                                         #
# ====================================================================================================================================== #

def gradient_descent_function(parameters, x_features, y_results, alfa):
    m = len(x_features)
    gradient_descent = list(parameters)
    for i in range(len(parameters)):
        temp = sum((hypothesis_function(parameters, x_features[j]) - y_results[j]) * x_features[j][i] for j in range(m))
        gradient_descent[i] = parameters[i] - alfa * (1/m) * temp
    return gradient_descent

# ====================================================================================================================================== #
#                                                       SCALING DATA SET                                                                 #
# ====================================================================================================================================== #

def scaling_function(x_features):
    x_features = np.asarray(x_features).T.tolist()
    for i in range(1, len(x_features)): 
        acum = sum(x_features[i])
        avg = acum / len(x_features[i])
        max_val = max(x_features[i])
        for j in range(len(x_features[i])):
            x_features[i][j] = (x_features[i][j] - avg) / max_val  # Mean scaling
    return np.asarray(x_features).T.tolist()

# ====================================================================================================================================== #
#                                                         1. Process Data Set                                                            #
# ====================================================================================================================================== #

temp_values = []
clouds_all_values = []
traffic_volume_values = []

with open('Metro_Interstate_Traffic_Volume.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        temp_values.append(float(row['temp']))
        clouds_all_values.append(float(row['clouds_all']))
        traffic_volume_values.append(float(row['traffic_volume']))

mean_temp = np.mean(temp_values)
mean_traffic_volume = np.mean(traffic_volume_values)
mean_clouds_all = np.mean(clouds_all_values)


x_features = []
y_results = []

with open('Metro_Interstate_Traffic_Volume.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        temp_value = float(row['temp'])
        clouds_value = float(row['clouds_all'])
        traffic_volume_value = float(row['traffic_volume'])

        if traffic_volume_value < 1000:
            continue

        if temp_value < 280 or temp_value > 300:
            temp_value = mean_temp
            traffic_volume_value = mean_traffic_volume
            clouds_value = mean_clouds_all

        x_features.append([1, temp_value, clouds_value])  # 1 for the bias term
        y_results.append(traffic_volume_value)

x_features = np.array(x_features)
y_results = np.array(y_results)

x_train, x_temp, y_train, y_temp = train_test_split(x_features, y_results, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

x_train_scaled = scaling_function(x_train)
x_val_scaled = scaling_function(x_val)
x_test_scaled = scaling_function(x_test)

# ====================================================================================================================================== #
#                                                              2. Computation                                                            #
# ====================================================================================================================================== #

parameters = [0, 0, 0]  # Initialize with zeros
alfa = 0.3
epoch = 300

train_errors = []
val_errors = []

for i in range(epoch):
    parameters = gradient_descent_function(parameters, x_train_scaled, y_train, alfa)
    train_error = mean_square_error_function(parameters, x_train_scaled, y_train)
    train_errors.append(train_error)
    val_error = mean_square_error_function(parameters, x_val_scaled, y_val)
    val_errors.append(val_error)
    print("Epoch %d: Training Error = %f, Validation Error = %f" % (i+1, train_error, val_error))

# ====================================================================================================================================== #
#                                                       Print Final Parameters                                                           #
# ====================================================================================================================================== #

print("Final Parameters (θs):")
for i, param in enumerate(parameters):
    print(f"θ{i}: {param}")

# ====================================================================================================================================== #
#                                                          Make a Prediction                                                             #
# ====================================================================================================================================== #

example_features = [1, 288, 40]  # Example: temp=mean_temp, clouds_all=40
example_features_scaled = scaling_function([example_features])[0]
predicted_value = hypothesis_function(parameters, example_features_scaled)
print(f"Predicted Traffic Volume: {predicted_value}")

# ====================================================================================================================================== #
#                                                      Calculate R-Squared Value                                                         #
# ====================================================================================================================================== #

def r_squared(y_real, y_pred):
    ss_total = sum((y_real - np.mean(y_real))**2)
    ss_residual = sum((y_real - y_pred)**2)
    return 1 - (ss_residual / ss_total)

y_train_pred = [hypothesis_function(parameters, x) for x in x_train_scaled]
y_val_pred = [hypothesis_function(parameters, x) for x in x_val_scaled]

train_r_squared = r_squared(y_train, y_train_pred)
val_r_squared = r_squared(y_val, y_val_pred)

print(f"R-Squared for Training Set: {train_r_squared}")
print(f"R-Squared for Validation Set: {val_r_squared}")

# ====================================================================================================================================== #
#                                                              3. Graph                                                                  #
# ====================================================================================================================================== #

plt.plot(range(epoch), train_errors, label='Training Error')
plt.plot(range(epoch), val_errors, label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.title('Training and Validation Error over Epochs')
plt.legend()
plt.grid(True)
plt.show()
