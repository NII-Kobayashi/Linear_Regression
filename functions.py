# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Functions for calculating the total number of retweets at the observation time and at the final time(s) of prediction.

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Szabo and Huberman, Communication of the ACM 53, pp.80-88, 2010;
Zhao et al., KDD, pp. 1513-1522, 2015*.
"""

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
    calculate the number of retweets at the observation time (=t_observation) and at the final time of prediction (=t_prediction)
    :param tweet_file: data file
    :param t_observation: observation time
    :param t_prediction: final time of prediction
    :param time_factor: factor to convert the time unit in seconds
    :return: tuple, the number of retweets at an observation time and at the final time
    """
    event_t_obs = 0
    event_t_pred = 0

    with open(tweet_file, "r") as in_file:
        first = next(in_file)  # to remove the first line in the data file
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
        return event_t_obs, event_t_pred  # not considering the original tweet


def no_of_events_in_window(event_file, t_hours, win_size, max_itr, time_factor=1):
    """
    calculate the number of retweets at the observation time and at the final times of prediction
    :param event_file: data file
    :param t_hours: observation time
    :param time_factor: factor to convert the time unit in seconds
    :param win_size: window size for prediction
    :param max_itr: the number of windows used in prediction
    :return: list, the number of retweets at an observation time and at the final times, and their logarithms
    """
    event_t = 0
    event_eof = 0
    event_eof_list = []
    event_eof_list_log = []
    time = t_hours * time_factor  # converting in seconds
    win_size_sec = win_size * time_factor
    t_f_list = [time + (i * win_size_sec) for i in range(1, max_itr + 1)]

    for i in range(len(t_f_list)):
        with open(event_file, "r") as in_file:
            first = next(in_file)  # to remove the first line containing total no of re-tweet and no of follower
            for num, line in enumerate(in_file, 0):
                # 0 to remove the original tweet, if first is used and 1 to add original tweet
                values = line.split(" ")
                if float(values[0]) <= time:
                    event_t = num
                if float(values[0]) <= (t_f_list[i]):
                    event_eof = num
            if event_t == 0:  # ignoring the file if there is no event happened during the observation time
                break
            else:
                event_eof_list.append(event_eof)
                event_eof_list_log.append(math.log(event_eof))

    if event_t == 0:  # ignoring the file if there is no event happened during till the observation time
        pass
    else:
        return event_eof_list, event_eof_list_log, event_t, math.log(event_t)  # not considering the original tweet


