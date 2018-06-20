# coding: utf-8
#PydevCodeAnalysisIgnore
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
    print '[INFO] DONE FILTERING DIRECTORY: {}'.format(path)


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
            'nrows', 'ncols', 'is_empty', 'pct_nan',
            'ts_fill_rate', 'ngaps']
    return df[cols]


def processing_dir(dir_path):
    # list_accepted = []
    alist = []
    clist = []
    list_dict = []
    pct_nan = 0.0
    ts_fill_rate = ''
    element = ''
    nrows = ''
    is_empty = False
    alert_level = ''

    for element in os.listdir(dir_path):
        csv_path = join(dir_path, element)
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

        c_message, alert_level, pct_nan = check_nan(df, 0.4)
        alist.append(alert_level)
        clist.append(c_message)

        alert_level = np.max(alist)
        if alert_level < 3:
            c_message, alert_level, gaps_list, ngaps = cu.check_gaps(df, 'B')
            alist.append(alert_level)
            clist.append(c_message)

            c_message, alert_level, ts_fill_rate = cu.check_fill_rate(df, 'B')
            alist.append(alert_level)
            clist.append(c_message)

        # control message
        if alert_level > 0:
            c_message = ', '.join([clist[i] for i, j in enumerate(clist) if j != 'OK'])
        var_dict = {'current_directory': basename(dir_path),
                    'var_name': element,
                    'nrows': nrows,
                    'ncols': ncols,
                    'is_empty': is_empty,
                    'pct_nan': pct_nan,
                    'ts_fill_rate': ts_fill_rate,
                    'alert': alert_level,
                    'ngaps': ngaps
                    }
        if alert_level != 0:
            # list_denied.append(element)
            # data_utils.write_dict_to_csv('Y.csv', var_dict, 'a')
            list_dict.append(var_dict)
    dfj = pd.DataFrame.from_dict(list_dict)
    dfj = reorder_df(dfj)
    return dfj


def main(path):
    df = pd.DataFrame()
    for pdir in os.listdir(path):
        c_dir = os.path.join(path, pdir)
        if os.path.isdir(c_dir):
            dfi = processing_dir(c_dir)
            df = df.append(dfi)
    df.to_csv('Files_to_check.csv')


path = '18 06 Derived Lab'
main(path)
