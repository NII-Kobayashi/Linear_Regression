# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Full example to checks the accuracy of the model.
Please replace file paths according to your local directory structure.

References
----------
.. *Szabo and Huberman, Communication of the ACM 53, 80 2010; Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
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


file_path = "Data/RT*.txt"
file_list_all = sorted(gb.glob(file_path), key=numerical_sort)
main(4, 6, file_list_all)
