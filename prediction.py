# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions for predicting the future re-tweets

References
----------
.. *Szabo and Huberman, Communication of the ACM 53, 80 2010; Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

from functions import *
import math


def event_prediction(alpha, var, r_t):
    """
    predict the total number of possible re-tweet at the time t
    :param alpha: estimator parameter for linear regression model
    :param var: estimator parameter for linear regression model
    :param r_t: the total number of tweet at the observation time
    :return: the biased estimator
    """

    if isinstance(alpha, float):
        r_est = r_t * (math.exp(alpha + (var / 2.0)))
        return r_est
    else:
        r_est = [r_t * (math.exp(alpha[i] + (var[i] / 2.0))) for i in range(len(alpha))]
        return r_est