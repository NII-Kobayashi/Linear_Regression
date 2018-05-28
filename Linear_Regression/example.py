from estimation import *
from prediction import *
import glob as gb
import numpy as np
import matplotlib.pyplot as plt

# estimation
T_OBS = 12
T_PRED = 16
file_list_training = sorted(gb.glob("../Data/training/RT*.txt"), key=numerical_sort)  # for all the training file
parameters_value = linear_regression_estimation(file_list_training, T_OBS, T_PRED)

# prediction
file_list_test = sorted(gb.glob("../Data/test/RT*.txt"), key=numerical_sort)  # for all the training file
nfile_prediction_result = linear_regression_prediction(parameters_value, T_OBS, T_PRED, file_list_test)
# print("The prediction result for n files are:", nfile_prediction_result)


# plot for one file data prediction
runtime = 15
event_list_test = [no_of_events(file_list_test[5], i, 6+i, 3600) for i in range(1, runtime +1)]
t_actual_value = [(event_list_test[i][0]) for i in range(len(event_list_test))]
t_pred_estimated = [event_prediction(parameters_value[0], parameters_value[1], event_list_test[i][0])
                    for i in range(6, runtime)]  # starting prediction from 6 hour (so i will get a value between 6-7)

y = np.arange(0, runtime, 1)
plt.step(y, np.append(np.array(t_actual_value[0]), np.diff(t_actual_value, 1)), linestyle="--")
plt.step(y[6:], np.append(np.array(t_pred_estimated[0]-t_actual_value[5]), np.diff(t_pred_estimated, 1)))
plt.yscale("log")
plt.xlabel('T (hours)')
plt.xlim(0, runtime + 1)
plt.ylabel('Events per window')
plt.title('Predicting twitter retweets')
