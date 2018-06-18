# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Functions for estimating the parameters of the linear regression model.

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Szabo and Huberman, Communication of the ACM 53, pp.80-88, 2010; Zhao et al., KDD, pp. 1513-1522, 2015*.
"""

from functions import *
import math


def parameters(log_r_inf, log_r_time):
    """
    Fit parameters of the linear regression model.
    :param log_r_time: array, the total number of retweets for each tweet at the observation time
    :param log_r_inf: array, the total number of retweets for each tweet at the prediction time
    :return: tuple, the linear regression model parameters (alpha and variance)
    """
    alpha = sum([(log_r_inf[i] - log_r_time[i]) for i in range(len(log_r_time))]) / len(log_r_time)
    var = sum([(log_r_inf[i] - log_r_time[i] - alpha) ** 2 for i in range(len(log_r_time))]) / len(log_r_time)
    return alpha, var


def linear_regression_estimation(training_data, t_obs, t_pred):
    """
    Return the estimated parameters if the number of training files are more than 10 (more than 10 tweets).
    Otherwise, an error message appears.
    :param training_data: array, the files used in training
    :param t_obs: the observation time
    :param t_pred: the prediction time
    :return: tuple, the linear regression model parameters (alpha and variance)
    """
    if len(training_data) < 10:
        raise Exception("There should be at-least 10 training files")
    event_list = [no_of_events(training_data[i], t_obs, t_pred, 3600) for i in range(len(training_data))]
    event_list = list(filter(None.__ne__, event_list))  # checking for none value
    log_t_obs = [math.log(event_list[i][0]) for i in range(len(event_list))]
    log_t_pred_actual = [math.log(event_list[i][1]) for i in range(len(event_list))]
    parameters_val = parameters(log_t_pred_actual, log_t_obs)
    return parameters_val
