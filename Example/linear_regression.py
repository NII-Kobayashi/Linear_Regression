from utilies_lr import *
import statistics
import matplotlib.pyplot as plt


def linear_regression_estimation(training_data, t_obs , t_pred):
    if len(training_data) < 10:
        raise Exception("There should be at-least 10 training file")
    event_list = [no_of_events(training_data[i], t_obs, t_pred, 3600) for i in range(len(training_data))]
    event_list = list(filter(None.__ne__, event_list))  # checking for none value
    log_t_obs = [math.log(event_list[i][0]) for i in range(len(event_list))]
    log_t_pred_actual = [math.log(event_list[i][1]) for i in range(len(event_list))]
    parameters_value = parameters_estimation(log_t_obs, log_t_pred_actual)
    return parameters_value


def linear_regression_prediction(parameters_estimated, t_obs, t_pred, test_data):
    event_list_test = [no_of_events(test_data[i], t_obs, t_pred, 3600) for i in range(len(test_data))]
    event_list_test = list(filter(None.__ne__, event_list_test))
    t_pred_estimated = [event_prediction(parameters_estimated[0], parameters_estimated[1], event_list_test[i][0])
                        for i in range(len(event_list_test))]
    t_actual_value = [event_list_test[i][1] for i in range(len(event_list_test))]
    # error = [abs(t_actual_value[i] - t_pred_estimated[i]) for i in range(len(t_actual_value))]
    # mean_error = statistics.mean(error)
    # print("The mean error is:", mean_error)
    # relative_error = sum(error) / sum(t_actual_value)
    # print("The relative error is:", relative_error)
    # plt.plot(t_actual_value)
    # plt.plot(t_pred_estimated)
    return t_pred_estimated



