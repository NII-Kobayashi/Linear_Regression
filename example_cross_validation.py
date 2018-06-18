"""
This code evaluates the linear regression model based on k-fold cross-validation.
Please replace file paths according to your local directory structure.

Inputs are
1) Data files that include the retweet times and the number of followers.
   Here, this code reads 'Data/RT*.txt' (= filename) and all the data files are saved in (= file_list_all)
2) Observation time (= obs_time_init).
3) Window size used in prediction (= window_size).
4) k_fold cross-validation (= k_fold)

Outputs is
1) Errors evaluated based on k_fold Cross-validation.
2) Plot of the mean error at different observation time

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""

from cross_validation import *
import glob as gb
import matplotlib.pyplot as plt

filename = "Data/RT*.txt"
file_list_all = sorted(gb.glob(filename), key=numerical_sort)
obs_time_init = 6  # observation time
window_size = 4  # window size for prediction
k_fold = 5  # k-fold iteration
mean_list = []  # save the mean value at different observation time
time_list = []  # save the different observation time considered
T_max = 168

for k in range(0, 5):
    if k == 4:
        obs_time = 72
    else:
        obs_time = obs_time_init * (2 ** k)
    no_of_bins = int((T_max - obs_time) / window_size)
    event_list = [no_of_events_in_window(file_list_all[i], obs_time, window_size, no_of_bins, 3600) for i in
                  range(len(file_list_all))]
    event_list = list(filter(None.__ne__, event_list))  # checking for none value
    result_lr = cross_validation_error(k_fold, event_list, no_of_bins)
    mean_list.append(result_lr[1])
    time_list.append(obs_time)
    print("Time:", obs_time, ", Median:", int(result_lr[0]), ", Mean:", int(result_lr[1]),
          ", Correlation (Median) = {0:.3f}".format(round(result_lr[2], 3)), ", Correlation (Mean) = {0:.3f}".format(round(result_lr[3], 3)))


# plot for mean error obtained at different observation time
plt.plot(time_list,  mean_list)
plt.xlabel('Observation Time (Hour)')
plt.ylabel('Mean absolute error')
plt.xticks(time_list)
plt.ylim(0)
plt.title("Mean Error plot for LR model")
plt.show()
