# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
function for evaluating the accuracy based on cross-validation

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Szabo and Huberman, Communication of the ACM 53, pp.80-88, 2010; Zhao et al., KDD, pp. 1513-1522, 2015*.
"""

import numpy as np
from sklearn.model_selection import KFold
import statistics
from estimation import *
from prediction import *


def cross_validation_error(k_fold, event_list_data, max_value_itr):
    """
    evaluate the mean and median of the errors, and their correlations
    :param k_fold: k_fold cross-validation
    :param event_list_data: list, the number of retweets at an observation time and at prediction times, and their logarithms
    :param max_value_itr: the number of windows for prediction
    :return: the mean and median of the errors and their correlations
    """
    parameters_value_list = []
    r_inf_estimated_list = []
    est_list_all = []
    actual_list_all = []
    error_list_all = []
    correlation_list = []
    kf = KFold(n_splits=k_fold)
    for train_index, test_index in kf.split(event_list_data):
        r_t_window = [(event_list_data[i][0]) for i in range(len(event_list_data))]
        r_t_window_log = [(event_list_data[i][1]) for i in range(len(event_list_data))]
        r_time = [(event_list_data[i][2]) for i in range(len(event_list_data))]

        r_time_log = [(event_list_data[i][3]) for i in range(len(event_list_data))]
        r_inf_array = np.asarray(r_t_window)
        r_inf_array_log = np.asarray(r_t_window_log)
        r_time_array = np.asarray(r_time)
        r_time_array_log = np.asarray(r_time_log)

        x_train_r_time_log, x_test_r_time_log = r_time_array_log[train_index], r_time_array_log[test_index]
        x_train_r_time, x_test_r_time = r_time_array[train_index], r_time_array[test_index]
        y_train_r_inf_log, y_test_r_inf_log = r_inf_array_log[train_index], r_inf_array_log[test_index]
        y_train_r_inf, y_test_r_inf = r_inf_array[train_index], r_inf_array[test_index]

        parameters_value = parameters(y_train_r_inf_log, x_train_r_time_log)
        parameters_value_list.append(parameters_value)

        # Prediction
        r_inf_estimated = [event_prediction(parameters_value[0], parameters_value[1], x_test_r_time[i])
                           for i in range(len(x_test_r_time))]
        r_inf_estimated_list.append(r_inf_estimated)

        # error estimation and correlation
        for i in range(len(r_inf_estimated)):
            value_estimated = r_inf_estimated[i]
            value_actual = y_test_r_inf[i]
            est_list = []
            actual_list = []
            s_xy_list = []
            s_x_list = []
            s_y_list = []
            correlation_val = 0
            for j in range(max_value_itr):
                if j == 0:
                    est_value = value_estimated[0] - x_test_r_time[i]
                    actual_value = value_actual[0] - x_test_r_time[i]
                    s_x = actual_value * actual_value
                    s_y = est_value * est_value
                    s_xy = actual_value * est_value
                else:
                    est_value = value_estimated[j] - value_estimated[j - 1]
                    actual_value = value_actual[j] - value_actual[j - 1]
                    s_x = actual_value * actual_value
                    s_y = est_value * est_value
                    s_xy = actual_value * est_value
                est_list.append(est_value)
                actual_list.append(actual_value)
                s_xy_list.append(s_xy)
                s_x_list.append(s_x)
                s_y_list.append(s_y)
            est_list_all.append(est_list)
            actual_list_all.append(actual_list)
            error_list = [abs(actual_list[j] - est_list[j]) for j in range(len(est_list))]  # error for one file
            error_list_all.append(sum(error_list))
            if sum(s_x_list) > 0 and sum(s_y_list) > 0:
                correlation_val = sum(s_xy_list) / math.sqrt(sum(s_x_list) * sum(s_y_list))
            if sum(s_x_list) > 0 and sum(s_y_list) == 0:
                correlation_val = 0
            if sum(s_x_list) == 0 and sum(s_y_list) > 0:
                correlation_val = 0
            if sum(s_x_list) == 0 and sum(s_y_list) == 0:
                correlation_val = 1
            correlation_list.append(correlation_val)

    med = statistics.median(error_list_all)
    mea = statistics.mean(error_list_all)
    med_cor = statistics.median(correlation_list)
    mea_cor = statistics.mean(correlation_list)
    return med, mea, med_cor, mea_cor
