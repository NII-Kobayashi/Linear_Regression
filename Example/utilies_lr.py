import re
import warnings
import math


numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    """
    numerically sort the filename path in the directory.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def no_of_events(tweet_file, t_observation, t_prediction, time_factor=1):
    """
    calculate the tweet number
    :param tweet_file: path to file
    :param t_observation: the observation time
    :param t_prediction: prediction time
    :param time_factor: factor to multiply time with, useful to convert time unit
    :return: the tuple containing the (total_no_of_tweets (at the specific time), tweets_at time_t)
    """
    event_t_obs = 0
    event_t_pred = 0

    with open(tweet_file, "r") as in_file:
        first = next(in_file)
        values_first = first.split(" ")
        for num, line in enumerate(in_file, 0):  # 0 to remove the original tweet  # change to 1 to add original tweet
            values = line.split(" ")
            if float(values[0]) <= (t_observation * time_factor):
                event_t_obs = num
            if float(values[0]) <= (t_prediction * time_factor):
                event_t_pred = num

    if event_t_obs == 0:
        warnings.warn("No event have occurred till the observation time. The file WILL BE IGNORED")
        print("Ignored File Name:", tweet_file)
    else:
        return event_t_obs, event_t_pred  # not considering the original tweet to count the original tweet


def parameters_estimation(log_r_time, log_r_pred):
    """
    calculate the parameters value of linear regression model
    :param log_r_time: array containing the total number of re-tweet at time T
    :param log_r_inf: array containing the value at the prediction time
    :return: the tuple containing the linear regression model parameters (alpha and variance)
    """
    alpha = sum([(log_r_pred[i] - log_r_time[i]) for i in range(len(log_r_time))]) / len(log_r_time)
    var = sum([(log_r_pred[i] - log_r_time[i] - alpha) ** 2 for i in range(len(log_r_time))]) / len(log_r_time)
    return alpha, var


def event_prediction(alpha, var, r_t):
    """
    estimate the total number of tweet
    :param alpha: alpha parameter value obtained from the data
    :param var: variance parameter value obtained from the data
    :param r_t: the time value for which we want to estimate the re-tweet number for each file
    :return: the biased estimator
    """
    r_est = r_t * (math.exp(alpha + (var / 2.0)))
    return r_est


