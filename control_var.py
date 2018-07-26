# coding: utf-8
# PydevCodeAnalysisIgnore
# Librairies
import pandas as pd
from os.path import join
import os
import control_utils as cu
import numpy as np
import logging
from var_logger import setup_logging
from os.path import basename
import shutil

# Launching logger
setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
logger.setLevel('DEBUG')


def filter_dir(path, denied_dir, flist):
    """
    Function
    Params
    """
    if not os.path.exists(denied_dir):
        os.makedirs(denied_dir)
    for element in os.listdir(path):
        for i, f in enumerate(flist):
            src = join(path, f)
            dst = join(denied_dir, f)
            if element == f:
                shutil.move(src, dst)
    logger.info('DONE FILTERING DIRECTORY: {}'.format(path))


def check_nan(df, pct):
    alert_level = 0
    c_message = 'OK'
    mis_val_percent = df.isnull().sum() / len(df)

    if mis_val_percent.values[0] > pct:
        alert_level = 2
        c_message = 'Many NaNs'

    return c_message, alert_level, mis_val_percent.values[0]


def reorder_df(df):
    cols = ['current_directory', 'var_name', 'alert',
            'nrows', 'ncols', 'is_empty', 'freq',
            'pct_nan',
            'ts_fill_rate', 'consecutive_nans',
            'longest_element', 'longest_repeat', 'c_message', 'ngaps']

    return df[cols]


def cons_nan_df(df, thr):
    """
    Consecutive NaN values
    """
    alert_level = 0
    c_message = 'OK'
    df2 = df.dropna()
    tdate = df2.index[0]
    dfx = df[df.index > tdate]
    nan_values = dfx[dfx.columns[0]].notnull().astype(int)
    nan_values = nan_values.cumsum()
    nan_count = dfx[dfx.columns[0]].isnull().astype(int)
    nan_count = nan_count.groupby(nan_values).sum()
    consecutive_nans = np.max(nan_count.values)
    if consecutive_nans > thr:
        alert_level = 2
        c_message = 'Many consecutive NaNs'
    return c_message, alert_level, consecutive_nans


def processing_dir(dir_path):
    # list_accepted = []
    list_dict = []
    element = ''
    for element in os.listdir(dir_path):
        alist = []
        clist = []
        alert_level = 0
        nrows = ''
        is_empty = False
        consecutive_nans = 0
        freq = ''
        long_rep = ''
        long_elem = ''
        c_message = ''
        pct_nan = 0.0
        ts_fill_rate = ''
        ngaps = 0
        csv_path = join(dir_path, element)
        print(element)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        c_message, alert_level, is_empty = cu.check_isempty(df)
        alist.append(alert_level)
        clist.append(c_message)

        c_message, alert_level, ncols = cu.check_ncols(df)
        alist.append(alert_level)
        clist.append(c_message)

        c_message, alert_level, nrows = cu.check_nrows(df)
        alist.append(alert_level)
        clist.append(c_message)

        if np.max(alist) != 3:
            c_message, alert_level, consecutive_nans = cons_nan_df(df, 1000)
            alist.append(alert_level)
            clist.append(c_message)
            logger.info('Consecutive NaN values:{} for {}'.
                        format(consecutive_nans, element))

            c_message, alert_level, pct_nan = check_nan(df, 0.4)
            alist.append(alert_level)
            clist.append(c_message)

        # Removing rows from a DataFrame which all values are NaN's
        df = cu.clean_rows_df(df)
        c_message, alert_level, nrows = cu.check_nrows(df)
        alist.append(alert_level)
        clist.append(c_message)

        c_message, alert_level, long_rep, long_elem = cu.check_mccv(df, 10)
        alist.append(alert_level)
        clist.append(c_message)

        alert = np.max(alist)
        print alist
        if alert != 3:
            freq = cu.infer_freq(df)
            if freq == pd.Timedelta('1 days'):
                freq = 'B'
            c_message, alert_level, ts_fill_rate = cu.check_fill_rate(df, freq)
            alist.append(alert_level)
            clist.append(c_message)

            c_message, alert_level, gaps_list, ngaps = cu.check_gaps(df, freq)
            alist.append(alert_level)
            clist.append(c_message)
        alert_level = np.max(alist)

        var_dict = {'current_directory': basename(dir_path),
                    'var_name': element,
                    'nrows': nrows,
                    'ncols': ncols,
                    'freq': freq,
                    'is_empty': is_empty,
                    'pct_nan': pct_nan,
                    'ts_fill_rate': ts_fill_rate,
                    'alert': np.max(alist),
                    'consecutive_nans': consecutive_nans,
                    'longest_element': long_elem,
                    'longest_repeat': long_rep,
                    'c_message': c_message,
                    'ngaps': ngaps
                    }
        print var_dict['alert']
        # control message
        if alert_level > 0:
            c_message = ', '.join([clist[i] for i, j in enumerate(clist) if j != 'OK'])

        #if alert_level != 0:
            # list_denied.append(element)
            # data_utils.write_dict_to_csv('Y.csv', var_dict, 'a')
        list_dict.append(var_dict)
    dff = pd.DataFrame.from_dict(list_dict)
    dff = reorder_df(dff)
    return dff


def main(path):

    df = pd.DataFrame()
    for pdir in os.listdir(path):
        logger.info('Processing directory: {}'.format(pdir))
        c_dir = os.path.join(path, pdir)
        if os.path.isdir(c_dir):
            if os.listdir(c_dir) != []:
                dfi = processing_dir(c_dir)
                df = df.append(dfi)

        df.to_csv('Dict_files.csv')

path = '1807 Derived'
#main('test')
#===============================================================================
main(path)
#===============================================================================
# df = processing_dir('1807 Derived/I')
# df.to_csv('Dict_files(III).csv')