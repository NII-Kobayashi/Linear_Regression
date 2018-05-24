from Linear_Regression.functions import *
import math


def event_prediction(alpha, var, r_t):
    """
    estimate the total number of possible tweet at the time t
    :param alpha: estimator parameter for linear regression model
    :param var: estimator parameter for linear regression model
    :param r_t: the total number of tweet at the observation time
    :return: the biased estimator
    """
    r_est = r_t * (math.exp(alpha + (var / 2.0)))
    return r_est


def linear_regression_prediction(parameters_estimated, t_obs, t_pred, test_data):
    """
    estimate the prediction value for a linear regression model
    :param parameters_estimated: estimated parameter (alpha, var) obtained for linear regression model
    :param t_obs: the observation time
    :param t_pred: the prediction time
    :param test_data: data files on which we want to calculate the prediction
    :return: the prediction value for all the files
    """
    event_list_test = [no_of_events(test_data[i], t_obs, t_pred, 3600) for i in range(len(test_data))]
    event_list_test = list(filter(None.__ne__, event_list_test))
    t_pred_estimated = [event_prediction(parameters_estimated[0], parameters_estimated[1], event_list_test[i][0])
                        for i in range(len(event_list_test))]
    return t_pred_estimated

