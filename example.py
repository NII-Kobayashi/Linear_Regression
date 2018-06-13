"""
This code trains the simple linear regression model parameters (alpha, variance) based on a re-tweet data-set
(data/training/RT*.txt), assuming the parameters are same in the data-set.
Please replace file paths according to your local directory structure.

Inputs are
1) Data file that includes the re-tweet times and the number of followers
Here, this code reads 'Data/training/RT*.txt' (= filename) and 'Data/test/RT*.txt' (= file_name_test) for test data set.
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).

Outputs is
1) The estimated parameters (alpha and variance)
2) The prediction result obtained form the model
3) The true prediction value
4) Error estimated

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""

from estimation import *
from prediction import *
import glob as gb
import numpy as np
import matplotlib.pyplot as plt

# estimation
T_OBS = 6
T_PRED = 10
filename = "Data/training/RT*.txt"  # path to the files used for training
file_list_training = sorted(gb.glob(filename), key=numerical_sort)  # for all the training file
parameters_value = linear_regression_estimation(file_list_training, T_OBS, T_PRED)

# prediction for one test file
file_name_test = "Data/test/RT2560.txt"  # path to the files used for prediction
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file
# prediction
event_list_test = no_of_events(file_name_test, T_OBS, T_PRED, 3600)
event_pred_true = event_list_test[1]
t_pred_estimated = event_prediction(parameters_value[0], parameters_value[1], event_list_test[0])
error = (abs(event_pred_true - t_pred_estimated))

print("The parameters estimated are:")
print("alpha :", parameters_value[0])
print("Variance:", parameters_value[1])
print("The prediction result for the observation time at ", T_OBS, "hours and the prediction time at", T_PRED, "hour is:")
print(t_pred_estimated)
print("The true value at the prediction time is", event_pred_true)
print("The error estimated is:", error)


