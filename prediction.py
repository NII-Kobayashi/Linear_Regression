# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Function for predicting the number of retweets in a future period

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Szabo and Huberman, Communication of the ACM 53, pp.80-88, 2010; Zhao et al., KDD, pp. 1513-1522, 2015*.
"""

from functions import *
import math


def event_prediction(alpha, var, r_t):
    """
     predict the total number of retweets
    :param alpha: a parameter of linear regression (alpha)
    :param var:  a parameter of linear regression (variance)
    :param r_t: the total number of tweet at the observation time
    :return: predicted number of retweets
    """

    if isinstance(alpha, float):
        r_est = r_t * (math.exp(alpha + (var / 2.0)))
        return r_est
    else:
        r_est = [r_t * (math.exp(alpha[i] + (var[i] / 2.0))) for i in range(len(alpha))]
        return r_est
