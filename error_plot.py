"""
This code trains the plot the mean error and the variance for the simple linear regression model
based on a re-tweet data-set (data/training/RT*.txt), assuming the parameters are same in the data-set.
Please replace file paths according to your local directory structure.

Inputs are
1) Data file that includes the re-tweet times and the number of followers
Here, this code reads 'Data/training/RT*.txt' (= filename) and 'Data/test/RT*.txt' (= file_name_test) for test data set.
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).
4) runtime for the T_OBS interval of 6 hours

Outputs is
1) the plot showing mean error and the variance.

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""

from estimation import *
from prediction import *
import glob as gb
import numpy as np
import matplotlib.pyplot as plt


def plot_different_time(t_obs, t_pred, file_list_train, file_list_test_):
    parameters_value = linear_regression_estimation(file_list_train, t_obs, t_pred)
    parameters_value = list(filter(None.__ne__, parameters_value))  # checking for none value

    # prediction
    event_list_test = [no_of_events(file_list_test_[i], t_obs, t_pred, 3600) for i in range(len(file_list_test_))]
    event_list_test = list(filter(None.__ne__, event_list_test))
    event_pred_true = [(event_list_test[i][1]) for i in range(len(event_list_test))]
    t_pred_estimated = [event_prediction(parameters_value[0], parameters_value[1], event_list_test[i][0])
                        for i in range(len(event_list_test))]

    error = [(abs(event_pred_true[i] - t_pred_estimated[i])) for i in range(len(event_pred_true))]
    mean_error = np.mean(error)
    sd = np.std(error)
    return mean_error, sd


filename = "Data/training/RT*.txt"
file_list = sorted(gb.glob(filename), key=numerical_sort)  # for files having tweet more than 20000 (RT186. RT1439)
file_name_test = "Data/test/RT*.txt"  # path to the files used for prediction
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file

# different observation time
T_OBS = 6
T_PRE = 78
runtime = 13
res = [plot_different_time((T_OBS*j), T_PRE, file_list, file_list_test) for j in range(1, runtime)]
mean_lr = [res[i][0] for i in range(len(res))]
std_lr = [res[i][1] for i in range(len(res))]
plt.errorbar(np.arange(6, 6*runtime, 6), np.log(mean_lr), np.log(std_lr), linestyle='None', marker='^', capsize=3)
plt.xlabel('T(hour) observation time')
plt.ylabel('Log Mean error')
plt.ylim(0)
plt.xticks(np.arange(6, 6*runtime, 6))
plt.title("Prediction value at T = 78 hours for different observation time")
plt.grid(True)
plt.show()
