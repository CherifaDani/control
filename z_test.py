from __future__ import division
import numpy as np
import data_utils
import pandas as pd
from dateutil import parser


# Librairies
import pandas as pd
import numpy as np
from os.path import sep
from dateutil import parser
from pandas.core.groupby import DataError
import logging
from var_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
logger.setLevel('DEBUG')


def check_gaps(df, freq):
    """
    Function checking if there are any gaps in a DF,
    according to its frequency


    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    freq : {Char type}
            The frequency of the DF

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case
    gaps_list : {List type}
                The list of gaps' dates

    n_gaps : {Int type}
            The number of detected gaps
    """
    c_message = 'OK'
    alert_level = 0
    gaps_list = []
    n_gaps = 0

    if freq:
        # print "[INFO] Original TS frequency : {}".format(freq)
        # Viewing the DataFrame as the full freq to make NaNs appear
        try:
            dff = df.resample(freq)
        except DataError:
            logger.exception('Impossible to resample the TS !..')
        else:
            # Filtering the DataFrame to only keep week days
            dff = dff[dff.index.dayofweek < 5]
            # Filtering again to only keep dates with NaNs
            dff = dff[dff.isnull().any(axis=1)]

            n_gaps = len(dff.index)

            if n_gaps > 0:
                logger.debug('Found {} missing values !..'.format(n_gaps))
                c_message = 'Gap(s) detected'
                alert_level = 3
                date_fmt = "%Y-%m-%d"
                dff.index = dff.index.format(formatter=lambda x: parser.
                                             parse(str(x)).strftime(date_fmt))
                gaps_list = dff.index.values

    return c_message, alert_level, gaps_list, n_gaps


df = data_utils.load_var('GOV.csv', 'GOV_JPN_1Y_Z250D')
c_message, alert_level, gaps_list, n_gaps = check_gaps(df, 'B')
