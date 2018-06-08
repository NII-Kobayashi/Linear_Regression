# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions for predicting the future re-tweets

References
----------
.. *Szabo and Huberman, Communication of the ACM 53, 80 2010; Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

from functions import *
import math


def event_prediction(alpha, var, r_t):
    """
    predict the total number of possible re-tweet at the time t
    :param alpha: estimator parameter for linear regression model
    :param var: estimator parameter for linear regression model
    :param r_t: the total number of tweet at the observation time
    :return: the biased estimator
    """

    if isinstance(alpha, float):
        r_est = r_t * (math.exp(alpha + (var / 2.0)))
        return r_est
    else:
        r_est = [r_t * (math.exp(alpha[i] + (var[i] / 2.0))) for i in range(len(alpha))]
        return r_est


def linear_regression_prediction(parameters_estimated, t_obs, t_pred, test_data):
    """
    call the event_prediction function to estimate the number of possible re-tweet at time t
    :param parameters_estimated: estimated parameter (alpha, var) obtained from linear_regression_estimation function
    :param t_obs: the observation time
    :param t_pred: the prediction time
    :param test_data: data files on which we want to do the prediction
    :return: the prediction value for all the files
    """
    event_list_test = [no_of_events(test_data[i], t_obs, t_pred, 3600) for i in range(len(test_data))]
    event_list_test = list(filter(None.__ne__, event_list_test))
    t_pred_estimated = [event_prediction(parameters_estimated[0], parameters_estimated[1], event_list_test[i][0])
                        for i in range(len(event_list_test))]
    return t_pred_estimated

