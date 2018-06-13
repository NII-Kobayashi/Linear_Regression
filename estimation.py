# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions for estimating the parameters (alpha and variance) used in simple linear regression model

References
----------
.. *Szabo and Huberman, Communication of the ACM 53, 80 2010; Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

from functions import *
import math


def parameters(log_r_inf, log_r_time):
    """
    calculate the parameters value for the linear regression model
    :param log_r_time: array containing the total number of re-tweet at time T
    :param log_r_inf: array containing the actual value of total number of re-tweet at the prediction time
    :return: the tuple containing the linear regression model parameters (alpha and variance)
    """
    alpha = sum([(log_r_inf[i] - log_r_time[i]) for i in range(len(log_r_time))]) / len(log_r_time)
    var = sum([(log_r_inf[i] - log_r_time[i] - alpha) ** 2 for i in range(len(log_r_time))]) / len(log_r_time)
    return alpha, var


def linear_regression_estimation(training_data, t_obs, t_pred):
    """
     call the no_of_events and parameters function and raise an exception if training files are less then 10
    :param training_data: the files used for training the model
    :param t_obs: the observation time
    :param t_pred: the prediction time
    :return: the tuple containing the linear regression model parameters (alpha and variance)
    """
    if len(training_data) < 10:
        raise Exception("There should be at-least 10 training files")
    event_list = [no_of_events(training_data[i], t_obs, t_pred, 3600) for i in range(len(training_data))]
    event_list = list(filter(None.__ne__, event_list))  # checking for none value
    log_t_obs = [math.log(event_list[i][0]) for i in range(len(event_list))]
    log_t_pred_actual = [math.log(event_list[i][1]) for i in range(len(event_list))]
    parameters_val = parameters(log_t_pred_actual, log_t_obs)
    return parameters_val
