from Linear_Regression.cross_validation import *
import glob as gb


def main(window_size, obs_time, file_list):
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
        return "Time:", t, result_lr


file_list_all = sorted(gb.glob("Data/RT*.txt"), key=numerical_sort)  # for all file
result_values = main(4, 6, file_list_all)
