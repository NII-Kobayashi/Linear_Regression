"""
This code evaluates the simple linear regression model parameters, based on a re-tweet data-set
(Data/training/RT*.txt), assuming the parameters are same in the data-set.
Please replace file paths according to your local directory structure.

Inputs are
1) Data file that includes the re-tweet times and the number of followers
Here, this code reads 'Data/RT*.txt' (= filename)
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).

Outputs is
1) Errors evaluated via Cross-Validation.

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""

from Linear_Regression.cross_validation import *
import glob as gb


def main(window_size, obs_time, file_list):
    """
       print the mean, median and correlation error at the observation
       :param window_size: the window size for multiple prediction value
       :param obs_time: observation time
       :param file_list: data files

       """
    for k in range(0, 5):
        if k == 4:
            t = 72
        else:
            t = obs_time * (2 ** k)
        max_value = int((168 - t) / 4)
        event_list = [no_of_events_in_window(file_list[i], t, window_size, max_value, 3600) for i in
                      range(len(file_list))]
        event_list = list(filter(None.__ne__, event_list))  # checking for none value
        result_lr = cross_validation_error(5, event_list, max_value)
        print("Time:", t, result_lr)


filename = "Data/RT*.txt"
file_list_all = sorted(gb.glob(filename), key=numerical_sort)
main(4, 6, file_list_all)
