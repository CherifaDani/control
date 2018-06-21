# coding: utf-8

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

"""
Author : N. MORIZET
Company: Advestis
Modified by: C. DANI
Last update: 29-05-2018
"""


def check_isempty(df):
    """
    Function checking if the dataframe is empty

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    Return
    ------
    c_message : {String type}
                The control message

    alert_level : {Int type}
                The alert level of each case

    is_empty : {Boolean type}
                Returns True if the Dataframe is empty
    """
    c_message = 'OK'
    alert_level = 0
    is_empty = False
    if df.empty:
        logger.warn('The Dataframe is empty !..')
        is_empty = True
        c_message = 'empty dataframe'
        alert_level = 3
    return c_message, alert_level, is_empty


def check_nrows(df):
    """
    Function checking the number of rows of a dataframe

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case

    nrows : {Int type}
            the number of rows
    """
    c_message = 'OK'
    alert_level = 0
    nrows = df.shape[0]
    if nrows < 2:
        logger.warn('The number of rows is less than 2 !..')
        c_message = 'nrows pb'
        alert_level = 3
    return c_message, alert_level, nrows


def check_ncols(df):
    """
    Function checking the number of columns of a dataframe

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe 

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case

    ncols : {Int type}
            the number of columns
    """
    c_message = 'OK'
    alert_level = 0
    ncols = df.shape[1]
    if ncols < 1:
        logger.warn('The number of columns is less than 2 !..')
        c_message = 'ncols pb'
        alert_level = 3
    return c_message, alert_level, ncols


def check_min_max(df, val_min, val_max):
    """
    Function checking if each value of the DF belongs to [min_value, max_value]

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    val_min : {float type}
              minimum value

    val_max : {float type}
                maximum value
    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case
    """

    c_message = 'OK'
    alert_level = 0
    ncols = df.shape[1]
    i = 0
    while i < ncols:
        val_vec = df.iloc[:, 0].values
        for val in val_vec:
            if val_min != '':
                if val < val_min:
                    c_message = 'min val pb'
                    alert_level = 2
                    break
            if val_max != '':
                if val > val_max:
                    c_message = 'max val pb'
                    alert_level = 2
                    break
        i += 1
    return c_message, alert_level


def longest_repetition(iterable):
    """
    Function returning .
    .
    Parameters
    ----------
    iterable : {Pandas series type}
                  Input series

    Return
    ------
    longest_element : {String type}
                        The item with the most consecutive repetitions in 'iterable',
                        If there are multiple such items, return the first one.
                        If 'iterable' is empty, return 'None'

    longest_repeats : {Int type}
                        The number of times the item is consecutively repeated
    """
    longest_element = current_element = None
    longest_repeats = current_repeats = 0
    for element in iterable:
        if current_element == element:
            current_repeats += 1
        else:
            current_element = element
            current_repeats = 1
        if current_repeats > longest_repeats:
            longest_repeats = current_repeats
            longest_element = current_element
    return longest_element, longest_repeats


# Function checking the maximum absolute change of the Time Series
def check_mac(df, val_mac):
    """
    Function checking the maximum absolute change of the dataframe

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    val_mac : {float type}
              The maximum absolute change value

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case
    max_over_val : {List type}
                    The list of values > val_mac
    """
    c_message = 'OK'
    alert_level = 0
    over_val = []
    max_over_val = []
    ncols = df.shape[1]
    i = 1  # i = 0 if index == True else i = 1
    while i < ncols:
        val_vec = df.iloc[:, i].values
        print val_vec

        if val_mac != '':
            # Relative Variation (note : take into account the frequency !)
            rel_var = np.abs(np.diff(val_vec)) / np.abs(val_vec[0:-1])
            over_val = rel_var[rel_var > val_mac]
            if len(over_val) > 0:
                c_message = 'max abs change overflow'
                alert_level = 3
                max_over_val.append(np.max(over_val))

            i += 1

    return c_message, alert_level, max_over_val


def check_mccv(df, max_ccv):
    """
    Function checking the maximum consecutive constant values

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    max_ccv : {Int type}
              the maximum consecutive constant values tolerable

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case
    """
    c_message = 'OK'
    alert_level = 0
    ncols = df.shape[1]
    i = 0
    while i < ncols:
        val_vec = df.iloc[:, i].values
        long_elem, long_rep = longest_repetition(val_vec)
        if long_rep > max_ccv:
            c_message = 'max ccv overflow'
            alert_level = 2
        i += 1
    return c_message, alert_level


def infer_freq(df):
    """
    Function inferring the frequency of a TS if not provided
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - freq : inferred frequency (type: character or timedelta)
    """
    freq = ''
    if 'Unnamed: 0' in list(df):  # Case for derived data ?
        df.set_index('Unnamed: 0', inplace=True)
        df.index.names = ['Date']
    df.index = pd.DatetimeIndex(df.index)
    # Sorting the dataframe in ascending order
    # df.sort(ascending=True, inplace=True)
    # Computing the gap distribution
    res = (pd.Series(df.index[1:]) - pd.Series(df.index[:-1])).value_counts()
    # print res
    if res.size != 0:
        freq = res.index[0]
    return freq


def check_fill_rate(df, freq):
    """
    Function computing the fill_rate of a DF from its first provided date

    Parameters
    ----------
    df : {Pandas dataframe type}
          Input dataframe

    freq : {Char type}
              The frequency of the TS, eg: 'B', 'D'

    Return
    ------
    c_message : {String type}
                 The control message

    alert_level : {Int type}
                    The alert level of each case
    
    ts_fill_rate: {Float type}
                    The fill rate of the time series
    """
    c_message = 'OK'
    alert_level = 0
    # Getting the current Timestamp
    ts_now = pd.tslib.Timestamp.now()
    # Retrieving the first Timestamp date
    ts_date = df.index[0]
    # Computing the range (in working days), between the two dates
    ts_range = pd.bdate_range(ts_date, ts_now, freq=freq)
    # Evaluating the fill rate
    ts_fill_rate = 1.0 * df.shape[0] / len(ts_range)
    if ts_fill_rate <= 0.9:
        print "[INFO] Fill rate = {}".format(ts_fill_rate)
        c_message = 'low fill rate <= {} % !'.format(ts_fill_rate)
        alert_level = 1
    return c_message, alert_level, ts_fill_rate


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
                alert_level = 2
                date_fmt = "%Y-%m-%d"
                dff.index = dff.index.format(formatter=lambda x: parser.
                                             parse(str(x)).strftime(date_fmt))
                gaps_list = dff.index.values

    return c_message, alert_level, gaps_list, n_gaps


def write_list_to_csv(flist, fname):
    """
    Function writing a list to a csv file

    Parameters
    ----------
    flist : {List type}
              Input list

    fname : {String type}
              The name of the csv file

    Return
    ------
    None
    """
    lf = len(flist)
    if lf > 0:
        df = pd.DataFrame(flist)
        # Formatting the csv file to save
        fname = fname.replace(sep, ' ')
        # Saving the DataFrame in a csv file
        df.to_csv(fname, sep=',')

