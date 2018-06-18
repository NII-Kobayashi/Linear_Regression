"""
This code trains the linear regression model by using a retweet dataaset (data/training/RT*.txt).
Please replace file paths according to your local directory structure.

Inputs are
1) Data files that include the retweet times and the number of followers.
   Here, this code reads 'Data/training/RT*.txt' (= filename) and 'Data/test/RT*.txt' (= file_name_test) for test data set.
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).

Outputs is
1) Estimated parameters (alpha and variance)
2) Predicted number of retweets from the observation time (= T_OBS) to the final time (= T_PRED).
3) Prediction error

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
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file more then one
# prediction
event_list_test = no_of_events(file_name_test, T_OBS, T_PRED, 3600)
event_pred_true = event_list_test[1]
t_pred_estimated = event_prediction(parameters_value[0], parameters_value[1], event_list_test[0])
error = (abs(event_pred_true - t_pred_estimated))

print("The parameters estimated are:")
print("alpha = {0:.3f}".format(round(parameters_value[0], 3)))
print("Variance = {0:.3f}".format(round(parameters_value[1], 3)))
print("Predicted number of retweets from the observation time (T_OBS=", T_OBS, "hours) to the final time (T_PRED=",
      T_PRED, "hours):", int(t_pred_estimated))
print("True value:", event_pred_true)
print("Prediction error:", int(error))
