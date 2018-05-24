from Linear_Regression.functions import *
import math


def linear_regression_estimation(training_data, t_obs, t_pred):
    """
    calculate the parameters value of linear regression model
    :param training_data: the files used for training the model
    :param t_obs: the observation time
    :param t_pred: the prediction time
    :return: the tuple containing the linear regression model parameters (alpha and variance)
    """
    if len(training_data) < 10:
        raise Exception("There should be at-least 10 training file")
    event_list = [no_of_events(training_data[i], t_obs, t_pred, 3600) for i in range(len(training_data))]
    event_list = list(filter(None.__ne__, event_list))  # checking for none value
    log_t_obs = [math.log(event_list[i][0]) for i in range(len(event_list))]
    log_t_pred_actual = [math.log(event_list[i][1]) for i in range(len(event_list))]
    alpha = sum([(log_t_pred_actual[i] - log_t_obs[i]) for i in range(len(log_t_obs))]) / len(log_t_obs)
    var = sum([(log_t_pred_actual[i] - log_t_obs[i] - alpha) ** 2 for i in range(len(log_t_obs))]) / len(log_t_obs)
    return alpha, var
