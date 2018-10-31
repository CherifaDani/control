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
from datetime import datetime
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
            'longest_element', 'longest_repeat', 'c_message', 'ngaps',
            'lag', 'start_date_mccv', 'end_date_mccv',
            'last_date']

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


def get_lag(df, freq, tdate=datetime.today()):
    tdate = pd.to_datetime('24/10/2018', dayfirst=True)

    refdays__m = pd.Timedelta('31 days')
    refdays__hebd = pd.Timedelta('7 days')
    refdays__q = pd.Timedelta('92 days')
    alert_level = 2
    last_update = df.index[-1]
    tdiff = tdate - last_update
    lag = tdiff.days
    if freq in ['B', 'D']:
        lag = tdiff.days
        return alert_level, lag
    elif freq == 'W' and tdiff < refdays__hebd:
        lag = tdiff.days
        return alert_level, lag
    elif freq == 'M' and tdiff < refdays__m:
        lag = tdiff.days
        return alert_level, lag
    elif freq == 'Q' and tdiff < refdays__q:
        lag = tdiff.days
        return alert_level, lag
    else:
        return alert_level, lag


def processing_dir(dir_path):
    # list_accepted = []
    list_dict = []
    element = ''
    last_date_mccv = None
    first_date_mccv = None

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
        last_date = None
        # tdate = pd.to_datetime('05/06/2018', dayfirst=True)
        lag = 0
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

        # if np.max(alist) != 3:
        #     c_message, alert_level, consecutive_nans = cons_nan_df(df, 1000)
        #     alist.append(alert_level)
        #     clist.append(c_message)
        #     logger.info('Consecutive NaN values:{} for {}'.
        #                 format(consecutive_nans, element))
        #
        #     c_message, alert_level, pct_nan = check_nan(df, 0.4)
        #     alist.append(alert_level)
        #     clist.append(c_message)

        # Removing rows from a DataFrame which all values are NaN's
        df = cu.clean_rows_df(df)
        c_message, alert_level, nrows = cu.check_nrows(df)
        alist.append(alert_level)
        clist.append(c_message)

        alert = np.max(alist)
        print alist
        if alert != 3:
            c_message, alert_level, long_rep, long_elem, start_date_mccv, end_date_mccv = cu.check_mccv(df, 10)
            alist.append(alert_level)
            clist.append(c_message)
            print alert_level
            last_date = df.index[-1]

            freq = cu.infer_freq(df)
            if freq == pd.Timedelta('1 days'):
                freq = 'B'
            elif freq == pd.Timedelta('7 days'):
                freq = 'W'
            elif freq == pd.Timedelta('31 days'):
                freq = 'M'
            elif freq == pd.Timedelta('92 days'):
                freq = 'Q'
            else:
                freq = freq.days
            print(freq)
            alert_level, lag = get_lag(df, freq=freq)
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
                    'ngaps': ngaps,
                    'lag': lag,
                    'last_date': last_date,
                    'start_date_mccv': start_date_mccv,
                    'end_date_mccv': end_date_mccv
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


# def main(path):
#
#     df = pd.DataFrame()
#     for pdir in os.listdir(path):
#         logger.info('Processing directory: {}'.format(pdir))
#         c_dir = os.path.join(path, pdir)
#         if os.path.isdir(c_dir):
#             if os.listdir(c_dir) != []:
#                 dfi = processing_dir(c_dir)
#                 df = df.append(dfi)
#
#         df.to_csv('Dict_files.csv')
def main(dir_path1, csv_name, diff_dir,  dirs=None):
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)

    df_global = pd.DataFrame()
    for element in dirs:
        rep1 = os.path.join(dir_path1, element)

        if len(os.listdir(rep1)) != 0:
            diff_dir_rep = os.path.join(diff_dir, rep1)
            if not os.path.exists(diff_dir_rep):
                os.makedirs(diff_dir_rep)
            dfi = processing_dir(rep1)
            df_global = df_global.append(dfi)
        df_global.to_csv(join(diff_dir, csv_name), index=False)


# path = '1807 Derived'
#main('test')
#===============================================================================
# main(path)
#===============================================================================
# df = processing_dir('1807 Derived/I')
# df.to_csv('Dict_files(III).csv')
# path = 'x.csv'
dirs = ['2 Data/2 Calculs/18 09 Derived/I',
        '2 Data/2 Calculs/18 09 Derived/AuxVariables',
        '2 Data/2 Calculs/18 09 Derived/Y',
        '2 Data/2 Calculs/18 09 Derived/X',
        '2 Data/1 Received/Market data/Base',
        ]

# path = '/home/cluster/MISSIONS/sesamm/test'
dir_path1 = '/media/HDD/MISSIONS_JUPYTER/compare_dirs_1v/Data_1V_Backtest_Analysis_241018'
# dirs = ['2 Data/1 Received/Market data/test']
print main(dir_path1, csv_name='control_1v-24.10.csv', diff_dir='1v', dirs=dirs)
# dfs = processing_dir(dir_path1)
# dfs.to_csv('sesamm.csv')


# df = pd.read_csv(path, index_col=0, parse_dates=True)
# df.index = pd.DatetimeIndex(df.index)
# df.sort_index(ascending=True, inplace=True)
# #
# freq = cu.infer_freq(df)
# # # print(freq)
# # # print(type(freq))
# # print(freq.days)
# tdate = pd.to_datetime('23/07/2018', dayfirst=True)
# # print(tdate)
# tdate = pd.to_datetime('01/06/2018', dayfirst=True)
#
# i = get_lag(df, freq=freq.days, tdate=tdate)
# print(i)
# # d = pd.Timedelta(days=3)
# # print d.days
