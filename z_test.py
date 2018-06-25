from __future__ import division
import numpy as np
import data_utils
import pandas as pd
from dateutil import parser
#import control_var as cv
from matplotlib import pyplot
import control_utils
# Librairies

import logging
from var_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
logger.setLevel('DEBUG')


# print df
# df.plot()
# pyplot.show()
def check_fill_rate(df, freq):
    """
    Function computing the fill_rate of a DF from its first provided date

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    freq : {Char type}
              The frequency of the TS, default: 'B'

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case

    ts_fill_rate : {Float type}
                    The fill rate of the time series
    """
    if freq == pd.Timedelta('1 days'):
        freq = 'B'

    c_message = 'OK'
    alert_level = 0
    ts_init = df.index[-1]
    # Retrieving the first Timestamp date
    ts_date = df.index[0]
    # Computing the range (in working days), between the two dates
    ts_range = pd.bdate_range(ts_date, ts_init, freq=freq)
    # Evaluating the fill rate
    ts_fill_rate = 1.0 * df.shape[0] / len(ts_range)
    if ts_fill_rate <= 0.9:
        print "[INFO] Fill rate = {}".format(ts_fill_rate)
        c_message = 'low fill rate <= {} % !'.format(ts_fill_rate)
        alert_level = 1
    return c_message, alert_level, ts_fill_rate

df = data_utils.load_var('ESTX600_FIN_EBIT1Y_RATIO_STD20_VS_STD100.csv', 'GOV_JPN_1Y_Z250D')
print check_fill_rate(df, 'B')
ts_init = df.index[-1]
# Retrieving the first Timestamp date
ts_date = df.index[0]
# Computing the range (in working days), between the two dates
ts_range = pd.bdate_range(ts_date, ts_init, freq='B')
print ts_range
print len(df)
