# coding: utf-8

# Librairies
import pandas as pd
import numpy as np
from os import listdir, getcwd
from os.path import join, isfile, sep
from dateutil import parser
import datetime
from workalendar.europe import France, UnitedKingdom, Germany, Greece, Italy, Spain
from workalendar.usa import UnitedStates
from workalendar.america import Brazil
from workalendar.asia import Japan
from pandas.core.groupby import DataError
from xlrd import XLRDError
import sys
sys.path.append(r'/home/ilies/git/Data-v2.0/')
sys.path.append(getcwd())
import data_utils as du


def files_from_dir(dir_name):
    """
    Function listing the files in a Directory
    :param dir_name: Directory name (type: string)
    :return:
        - files_list : files in the Directory (type: list)
        - nfiles : number of files on the Directory (type: int)
    """
    nfiles = 0
    files_list = [x for x in listdir(dir_name) if isfile(join(dir_name, x)) and not x.startswith('.')]
    files_list = sorted(files_list)
    if files_list == []:
        print "[INFO] The '{}' Directory is empty !..".format(dir_name)
    else:
        nfiles = len(files_list)
        print "[INFO] Processing '{}' Directory".format(dir_name)
        print "[INFO] Number of files : '{}'\n".format(nfiles)
    return files_list, nfiles


def alterate_base_dir(base_directory):
    """
    Function alterating the original 'Base' directory to simulate non up-to-date csv files
    :param base_directory: base directory path (type: string)
    :return: Nothing
    """
    blist = [x for x in listdir(base_directory) if isfile(join(base_directory, x)) and not x.startswith('.')]
    blist = sorted(blist)
    for f in blist:
        # for f in blist[0:1]:  # TEST PURPOSE !!
        # print "[INFO] Alterating file '{}' in '{}' directory..".format(f, base_directory)
        df = pd.read_csv(join(base_directory, f))
        # Choosing a random integer between 1 and 6
        n = np.random.randint(low=1, high=7, size=1)[0]
        # Dropping the n last rows from the DataFrame
        dff = df.drop(df.tail(n).index)
        # Indexing
        dff.set_index('Date', inplace=True)
        # Saving the altered DataFrame to disk
        dff.to_csv(join(base_directory, f), sep=',')
    print "[INFO] Alterating files in '{}' directory COMPLETE !".format(base_directory)


def primary_checks(bdir, ldir):
    """
    Function executing basic checks for the main directories
    :param bdir: base directory path (type: string)
    :param ldir: latest data directory path (type: string)
    :return:
        - bdir_list: list of files in base directory (type: list)
        - ldir_list: list of files in latest data directory (type: list)
        - list_bnotinl: list of files in base directory NOT in latest data directory (type: list)
        - list_lnotinb: list of files in latest data directory NOT in base directory (type: list)
        - list_common: list of common files from both base and latest data directories (type: list)
    """
    bdir_list = [x for x in listdir(bdir) if isfile(join(bdir, x)) and not x.startswith('.')]
    bdir_list = sorted(bdir_list)
    ldir_list = [x for x in listdir(ldir) if isfile(join(ldir, x)) and not x.startswith('.')]
    ldir_list = sorted(ldir_list)
    list_bnotinl = []
    list_lnotinb = []
    list_common = []
    if not ldir_list:
        print "[INFO] The '{}' Directory is empty ! No updates to perform !..".format(ldir)
    else:
        nbdir = len(bdir_list)  # Number of files inside the 'Base' directory
        nldir = len(ldir_list)  # Number of files inside the 'Latest data' directory
        print "[INFO] '{}' Directory : {} files".format(bdir, nbdir)
        print "[INFO] '{}' Directory : {} files".format(ldir, nldir)
        # Checking if some elements from 'Base' are not in 'Latest data'
        list_bnotinl = [x for x in bdir_list if x not in ldir_list]
        list_bnotinl = sorted(list_bnotinl)
        # Checking if some elements from 'Latest data' are not in 'Base'
        list_lnotinb = [x for x in ldir_list if x not in bdir_list]
        list_lnotinb = sorted(list_lnotinb)
        # Checking common elements from both 'Latest data' and 'Base' directories
        list_common = [x for x in bdir_list if x in ldir_list]
        list_common = sorted(list_common)
        # List sizes
        len_bnotinl = len(list_bnotinl)
        len_lnotinb = len(list_lnotinb)
        len_common = len(list_common)
        if len_bnotinl > 0:
            print "[INFO] List of elements from 'Latest data' not in 'Base' : {}".format(list_bnotinl)
        if len_lnotinb > 0:
            print "[INFO] List of elements from 'Base' not in 'Latest data' : {}".format(list_lnotinb)
        if len_common > 0:
            print "[INFO] Number of common elements from 'Base' and 'Latest data' : {}".format(len_common)
        print "[INFO] Primary checks OK !\n"
    return bdir_list, ldir_list, list_bnotinl, list_lnotinb, list_common


def read_deriv_script(filename):
    """
    Function reading the derivation script from the filename and generating a DataFrame
    :param filename: derivation script filename (type: string)
    :return:
        - dfs: Dataframe of the read file (type: pandas DataFrame)
        - sheets : excel sheet names (type : list)
    """
    dfs = pd.DataFrame()
    sheetname = 'primary'
    # Attempt to extract the sheet name without loading the whole file
    xls = pd.ExcelFile(filename, on_demand=True)
    sheets = xls.sheet_names
    if 'Primary' in sheets:
        sheetname = 'Primary'
    try:
        dfs = pd.read_excel(filename, sheetname=sheetname)
    except IOError:
        print '[ERROR] Impossible to read the derivation script !..'
    except XLRDError:
        print "[ERROR] No sheet named <'{}'> !".format(sheetname)
    else:
        # Converting 'SQL_Name' column to uppercase
        dfs['SQL_Name'] = map(lambda x: x.upper(), dfs['SQL_Name'])
        print '[INFO] Derivation script read > OK !\n'
    return dfs, sheets


def gen_activated_list(dfs, sheets):
    """
    Function giving the 'activated' csv files list from the Derivation Script (primary)
    :param dfs: Derivation script DataFrame (type: pandas DataFrame)
    :param sheets: sheet names from the Derivation Script (type: list)
    :return:
        - activated_list_names: csv files to take into account for the processes (type: list)
    """
    dfs_active_col = 'ACTIVE'
    if 'Active' in sheets:
        dfs_active_col = 'Active'
    activated_list = dfs.loc[dfs[dfs_active_col] == 1]['Path'].values
    activated_list_names = map(lambda x: x.split(sep)[-1], activated_list)
    return activated_list_names


def try_find_delim(csv_file):
    """
    Function trying to find the delimiter of a csv file
    :param csv_file: csv file (type: string)
    :return:
        - df: Dataframe of the read file (type: pandas DataFrame)
        - delim: csv file separator (type: character)
        - integrity_message: integrity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
    """
    integrity_message = 'OK'
    alert_level = 0
    df = pd.DataFrame()
    # Sniffing the delimiter used in the csv file (using data-v2.0)
    delimiters = ',;'
    delim = du.find_delim(csv_file, delimiters=delimiters)
    if delim not in delimiters:
        print '[ERROR] Delimiter is not consistent !..'
        integrity_message = 'delimiter pb'
        alert_level = 1
    else:
        df = pd.read_csv(csv_file, sep=delim)
    return df, delim, integrity_message, alert_level


def try_read_csv(csv_file):
    """
    Function trying to read a csv file
    :param csv_file: csv file (type: string)
    :return:
        - integrity_message: integrity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - is_readable: is the file readable or not (type: bool)
    """
    integrity_message = 'OK'
    alert_level = 0
    is_readable = False
    try:
        pd.read_csv(csv_file)
    except IOError:
        print '[ERROR] Impossible to read the csv file !..'
        integrity_message = 'read pb'
        alert_level = 3
    else:
        is_readable = True
    return integrity_message, alert_level, is_readable


def check_isempty(df):
    """
    Function checking if the DataFrame is empty
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - integrity_message: integrity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - is_empty: is the DataFrame empty or not (type: bool)
    """
    integrity_message = 'OK'
    alert_level = 0
    is_empty = False
    if df.empty:
        print '[ERROR] The Dataframe is empty !..'
        is_empty = True
        integrity_message = 'empty dataframe'
        alert_level = 3
    return integrity_message, alert_level, is_empty


def check_ncols(df):
    """
    Function checking the number of columns in a pandas DataFrame
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - integrity_message: integrity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - ncols: number of columns (type: int)
    """
    integrity_message = 'OK'
    alert_level = 0
    # nrows = df.shape[0]
    ncols = len(df.columns)
    if ncols != 2:
        print '[ERROR] The number of columns is not 2 !..'
        integrity_message = 'ncols pb'
        alert_level = 3
    return integrity_message, alert_level, ncols


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


def fill_dict_from_df(df, dfs):
    """
    Function filling the dictionnary containing information for a single input DataFrame
    :param df: Input DataFrame (type: pandas DataFrame)
    :param dfs: Derivation script lsit (type: list)
    :return:
        - cat_name: category name (type: string)
        - sql_name: SQL Name (type: string)
        - freq: frequency (type: character)
        - country: benchmark country (type: string)
        - min_val: minimum value (type: float)
        - max_val: maximum value (type: float)
        - mac_val: maximum absolute change value (type: float)
        - mcmv_val: maximum consecutive missing values (type: int)
        - mccv_val: maximum consecutive constant values (type: int)
    """
    # Initialization
    cat_name = ''
    sql_name = ''
    country = ''
    min_val = ''
    max_val = ''
    mac_val = ''
    mcmv_val = ''
    mccv_val = ''

    if len(dfs) > 0:
        sql_name = list(df)[-1]
        # Retrieving row information from 'SQL_Name'
        sel_row = dfs.loc[dfs['SQL_Name'] == sql_name]
        # Category
        cat_name = sel_row['Rubrique'].values
        cat_name = cat_name[0].encode("utf-8") if cat_name.size != 0 and not pd.isnull(cat_name) else ''
        # SQL Name
        sql_name = sel_row['SQL_Name'].values
        sql_name = sql_name[0] if sql_name.size != 0 else ''
        # Frequency
        # Inferring the frequency from the TS if not provided !
        freq = sel_row['Frequence'].values
        freq = infer_freq(df) if not freq else str(freq[0])
        # if not freq:
        #     # print "[WARNING] Frequency NOT provided, inferring it !.."
        #     freq = infer_freq(df)
        # else:
        #     freq = str(freq[0])
        #     # print "[INFO] Frequency provided : {}".format(freq)
        # Benchmark Country
        country = sel_row['benchmark_country'].values
        country = country[0].encode("utf-8") if country.size != 0 and not pd.isnull(country) else ''
        # Minimum Value
        min_val = sel_row['Minimum Value'].values
        min_val = np.float(min_val[0]) if min_val.size != 0 and not pd.isnull(min_val) else ''
        # Maximum Value
        max_val = sel_row['Maximum value'].values
        max_val = np.float(max_val[0]) if max_val.size != 0 and not pd.isnull(max_val) else ''
        # Maximum Absolute Change
        mac_val = sel_row['maximum absolute change'].values
        mac_val = np.float(mac_val[0]) if mac_val.size != 0 and not pd.isnull(mac_val) else ''
        # Maximum Consecutive Missing Values
        mcmv_val = sel_row['Max consecutive missing values'].values
        mcmv_val = np.int(mcmv_val[0]) if mcmv_val.size != 0 and not pd.isnull(mcmv_val) else ''
        # Maximum Consecutive Constant Values
        mccv_val = sel_row['Max consecutive constant values'].values
        mccv_val = np.int(mccv_val[0]) if mccv_val.size != 0 and not pd.isnull(mccv_val) else ''
    else:
        freq = infer_freq(df)

    return cat_name, sql_name, freq, country, min_val, max_val, mac_val, mcmv_val, mccv_val


def check_date_col(df):
    """
    Function checking whether or not the DataFrame contains a 'Date' column
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - integrity_message: integrity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - has_date_col: has the DataFrame a date column or not (type: bool)
    """
    integrity_message = 'OK'
    alert_level = 0
    has_date_col = False
    if 'Date' in list(df):
        has_date_col = True
    else:
        # For the Derived data, the first date column is in Index, we try to rebuild that column !
        df.set_index('Unnamed: 0', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        if not isinstance(df.index, pd.DatetimeIndex):
            print "[WARNING] No 'Date' column in the Dataframe !.."
            integrity_message = 'no date col'
            alert_level = 3
    return integrity_message, alert_level, has_date_col


def check_col_nans(df):
    """
    Function checking if there is a column full of NaN's
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - integrity_message: integrity message (type: string)
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
    """
    integrity_message = 'OK'
    validity_message = 'OK'
    alert_level = 0
    nrows = df.shape[0]
    ncols = len(df.columns)
    for i in range(ncols):
        nan_sum = df.isnull().sum(axis=0)[i]
        if nan_sum == nrows:
            integrity_message = 'only NaNs'
            validity_message = 'only NaNs'
            alert_level = 3
            break
    return integrity_message, validity_message, alert_level


def clean_rows_df(df):
    """
    Function removing rows from a DataFrame which all values are NaN's
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - dff : filtered DataFrame (type: pandas DataFrame)
    """
    dff = df.dropna(how='any', axis=0)
    return dff


def check_filt_nrows(df):
    """
    Function checking the number of rows in a filtered DataFrame
    :param df: Input filtered DataFrame (type: pandas DataFrame)
    :return:
        - integrity_message: integrity message (type: string)
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - nrows: number of remaining rows after filtering
    """
    integrity_message = 'OK'
    validity_message = 'OK'
    alert_level = 0
    nrows = df.shape[0]
    if nrows == 0:
        integrity_message = 'no rows to update'
        validity_message = 'no rows to update'
        alert_level = 3
    return integrity_message, validity_message, alert_level, nrows


def is_today_date_included(df):
    """
    Function checking whether or not today's date is included in the filtered DataFrame
    :param df: Input filtered DataFrame (type: pandas DataFrame)
    :return:
        - df: date-normalized / indexed (type: pandas DataFrame)
    """
    has_today_date = False
    # Date normalization format
    date_fmt = "%Y-%m-%d"
    # Retrieving today's date
    now = datetime.datetime.now()
    today_date = now.strftime(date_fmt)
    # today_date = "2000-01-04"  # TEST PURPOSE !!
    # Normalizing the date column format
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
        df.index = df.index.format(formatter=lambda x: parser.parse(str(x)).strftime(date_fmt))
        if today_date in df.index:
            has_today_date = True
    # Renaming the index
    df.index.names = ['Date']
    return df, has_today_date


def longest_repetition(iterable):
    """
    Function returning the item with the most consecutive repetitions in 'iterable'.
    If there are multiple such items, return the first one.
    If 'iterable' is empty, return 'None'.
    :return:
        - longest_element: element with the longest repeats
        - longest_repeats: number of longest repeats
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


def check_min_max(df, val_min, val_max):
    """
    Function checking if each value of the TS belongs to [min_value, max_value]
    :param df: Input DataFrame (type: pandas DataFrame)
    :param val_min: minimum value (type: float)
    :param val_max: maximum value (type: float)
    :return:
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
    """
    validity_message = 'OK'
    alert_level = 0
    val_vec = df.iloc[:, 0].values
    for val in val_vec:
        if val_min != '':
            if val < val_min:
                validity_message = 'min val pb'
                alert_level = 2
                break
        if val_max != '':
            if val > val_max:
                validity_message = 'max val pb'
                alert_level = 2
                break
    return validity_message, alert_level


def check_mac(df, val_mac):
    """
    Function checking the maximum absolute change of the TS
    :param df: Input DataFrame (type: pandas DataFrame)
    :param val_mac: maximum absolute change value (type: float)
    :return:
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - max_over_val: maximum value from those above the mac threshold
    """
    validity_message = 'OK'
    alert_level = 0
    max_over_val = []
    val_vec = df.iloc[:, 0].values
    if val_mac != '':
        # Relative Variation (note : take into account the frequency !)
        rel_var = np.abs(np.diff(val_vec)) / np.abs(val_vec[0:-1])
        over_val = rel_var[rel_var > val_mac]
        if len(over_val) > 0:
            validity_message = 'max abs change overflow'
            alert_level = 3
            max_over_val = np.max(over_val)
    return validity_message, alert_level, max_over_val


def is_access_denied(df):
    """
    Function addressing the 'Access Denied' issue in a filtered DataFrame
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - integrity_message: integrity message (type: string)
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
    """
    integrity_message = 'OK'
    validity_message = 'OK'
    alert_level = 0
    x_val = df.iloc[:, 0].values
    if any('Access Denied' in str(s) for s in x_val):
        integrity_message = 'access denied'
        validity_message = 'access denied'
        alert_level = 3
    return integrity_message, validity_message, alert_level


def check_mccv(df, max_ccv):
    """
    Function checking the maximum consecutive constant values
    :param df: Input DataFrame (type: pandas DataFrame)
    :param max_ccv:
    :return:
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
    """
    validity_message = 'OK'
    alert_level = 0
    val_vec = df.iloc[:, 0].values
    long_elem, long_rep = longest_repetition(val_vec)
    if long_rep > max_ccv:
        validity_message = 'max ccv overflow'
        alert_level = 2
    return validity_message, alert_level


def check_fill_rate(df, freq, thr_fr):
    """
    Function computing the fill_rate of a Time Series from its first provided date
    :param df: Input DataFrame (type: pandas DataFrame)
    :param freq: Frequency of the Time Series
    :param thr_fr: fill rate threshold
    :return:
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - ts_fill_rate: fill rate of the Time Series (type: float) €[0,1]
    """
    validity_message = 'OK'
    alert_level = 0
    # Getting the current Timestamp
    ts_now = pd.tslib.Timestamp.now()
    # Retrieving the first Timestamp date
    ts_date = df.index[0]
    # Computing the range (in working days), between the two dates
    # ts_range = pd.bdate_range(ts_date, ts_now, freq=freq)
    ts_range = pd.bdate_range(ts_date, ts_now)
    # Evaluating the fill rate
    ts_fill_rate = 1.0 * df.shape[0] / len(ts_range)
    if ts_fill_rate <= thr_fr:
        # print "[ERROR] Fill rate = {}".format(ts_fill_rate)
        validity_message = 'low fill rate <= {} !'.format(thr_fr)
        alert_level = 3
    return validity_message, alert_level, ts_fill_rate


def integrity_check(data_dir, csv_file):
    """
    Function checking the integrity of a TS (csv file)
    :param data_dir: data directory (base, latest or derived) (type: string)
    :param csv_file: csv filename (type: string)
    :return:
        - df: Filtered DataFrame (type: pandas DataFrame)
        - df_info_dict: DataFrame information dictionnary (type: dict)
    """

    # Initialization
    ilist = []  # Integrity messages list
    vlist = []  # Validity messages list
    alist = []  # Alert level list

    # Renaming the input csv file to fullname
    csv_file = join(data_dir, csv_file)

    # Initializing the dictionnary info
    df_info_dict = {'category': '',
                    'sql_name': '',
                    'filename': csv_file,
                    'readable': False,
                    'delimiter': '',
                    'is_empty': False,
                    'ncols': 0,
                    'has_date_col': False,
                    'nrows_after_filter': 0,
                    'today_date_included': False,
                    'frequency': '',
                    'benchmark_country': '',
                    'min_value': '',
                    'max_value': '',
                    'max_abs_change': '',
                    'max_mac_values': [],
                    'max_cmv': '',
                    'max_ccv': '',
                    'ngaps': 0,
                    'gap_dates': [],
                    'fill_rate': 0,
                    'integrity_check': 'OK',
                    'validity_check': 'OK',
                    'alert_level': 0}

    # Use list of functions and **kwargs ?
    # https://pythontips.com/2013/08/04/args-and-kwargs-in-python-explained/

    # Checking if the csv file is readable
    integrity_message, alert_level, readable = try_read_csv(csv_file)
    df_info_dict['readable'] = readable
    ilist.append(integrity_message)
    alist.append(alert_level)

    # Finding the delimiter in the csv file
    df, delim, integrity_message, alert_level = try_find_delim(csv_file)
    df_info_dict['delimiter'] = "'%s'" % delim
    ilist.append(integrity_message)
    alist.append(alert_level)

    # Checking if the DataFrame is empty
    integrity_message, alert_level, isempty = check_isempty(df)
    df_info_dict['is_empty'] = isempty
    ilist.append(integrity_message)
    alist.append(alert_level)

    # Checking the number of columns of the DataFrame
    integrity_message, alert_level, ncols = check_ncols(df)
    df_info_dict['ncols'] = ncols
    ilist.append(integrity_message)
    alist.append(alert_level)

    # Checking if there is a column full of NaN's
    integrity_message, validity_message, alert_level = check_col_nans(df)
    ilist.append(integrity_message)
    vlist.append(validity_message)
    alist.append(alert_level)

    # # Checking whether or not there is a 'Date' column
    # integrity_message, alert_level, has_date_col = check_date_col(df)
    # df_info_dict['has_date_col'] = has_date_col
    # ilist.append(integrity_message)
    # alist.append(alert_level)

    # Removing index duplicates
    df = df.groupby(df.index).first()

    # Sorting by index, ascending order
    df.sort(ascending=True, inplace=True)

    # Removing rows from a DataFrame which all values are NaN's
    df = clean_rows_df(df)

    # Checking the number of rows in the filtered DataFrame
    integrity_message, validity_message, alert_level, filt_nrows = check_filt_nrows(df)
    df_info_dict['nrows_after_filter'] = filt_nrows
    ilist.append(integrity_message)
    vlist.append(validity_message)
    alist.append(alert_level)

    # Addressing the 'Access Denied' issue
    integrity_message, validity_message, alert_level = is_access_denied(df)
    ilist.append(integrity_message)
    vlist.append(validity_message)
    alist.append(alert_level)

    # Checking whether or not today's date is included
    df, today_date = is_today_date_included(df)
    df_info_dict['today_date_included'] = today_date

    # Alert Level
    alert_level = np.max(alist)
    df_info_dict['alert_level'] = alert_level
    # print 'alert_level = {}'.format(alert_level)

    # Integrity Message & Validity Message
    if alert_level > 0:
        # print df_info_dict['sql_name']
        integrity_message = ', '.join([ilist[i] for i, j in enumerate(ilist) if j != 'OK'])
        df_info_dict['integrity_check'] = integrity_message
        # print integrity_message
        validity_message = ', '.join([vlist[i] for i, j in enumerate(vlist) if j != 'OK'])
        df_info_dict['validity_check'] = validity_message

    return df, df_info_dict


def get_cal_from_country(bmk_country):
    """
    Function returning a calendar based on the 'benchmark_country' of the csv file
    # Python package to manage holidays per country
    # >> See : https://github.com/peopledoc/workalendar
    from 'benchmark_country' column (to be parsed) in the Derivation Script
    Warning : Tuples may appear like [USA, Japon] or [USA, China] instead of China
    [Germany, China] instead of Russia
    [USA, China] instead of Moyen-Orient
    [USA, China] instead of Brasil
    Currently missing : China, Russia
    @TODO : ADD HONG-KONG !!! (for 'HSI_Index')
    NOTE :  5 avril 2018 : Ching Ming Festival (jour férié Hong-Kong !)
    :param bmk_country: benchmark country (type: string)
    :return:
        - cal: calendar related to the country (type: workalendar type ?)
    """
    cal = []
    if ',' in bmk_country:  # '[A, B]' => ['A', 'B']
        print "[WARNING] Tuple for the 'benchmark_country : {}, returning the first one..".format(bmk_country)
        bmk_country = bmk_country.replace('[', '').replace(']', '').split(',')
        bmk_country = bmk_country[0]  # TO BE DEFINED !

    if bmk_country == 'USA':
        cal = UnitedStates()
    elif bmk_country == 'Germany':
        cal = Germany()
    elif bmk_country == 'Japan':
        cal = Japan()
    elif bmk_country == 'France':
        cal = France()
    elif bmk_country == 'UK':
        cal = UnitedKingdom()
    elif bmk_country == 'Grèce':
        cal = Greece()
    elif bmk_country == 'Italie':
        cal = Italy()
    elif bmk_country == 'Espagne':
        cal = Spain()
    elif bmk_country == 'Brasil':
        cal = Brazil()
    return cal


def check_gaps(df, freq, bmk_country):
    """
    Function checking if there are any gaps in a TS
    :param df: Input DataFrame (type: pandas DataFrame)
    :param freq: TS frequency (type: character or timedelta)
    :param bmk_country: benchmark country (type: string)
    :return:
        - validity_message: validity message (type: string)
        - alert_level: alert level [0; 3] (type: int)
        - gaps_list: list of found missing values (type: list)
    """
    validity_message = 'OK'
    alert_level = 0
    gaps_list = []
    n_gaps = 0

    # Converting 'Date' column to 'DatetimeIndex'
    # df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)

    if freq:
        # print "[INFO] Original TS frequency : {}".format(freq)
        # Viewing the DataFrame as the full freq to make NaNs appear
        try:
            dff = df.resample(freq)
        except DataError:
            print "[ERROR] Impossible to resample the TS !.."
        else:
            # Filtering the DataFrame to only keep week days
            dff = dff[dff.index.dayofweek < 5]
            # Filtering again to only keep dates with NaNs
            dff = dff[dff.isnull().any(axis=1)]
            if bmk_country:
                # print "[INFO] Original TS benchmark_country : {}".format(bmk_country)
                # Filtering the holidays (federal only ?) according to the country !
                cal = get_cal_from_country(bmk_country)  # type: list
                if cal:
                    dff = dff.iloc[map(lambda x: cal.is_working_day(x.date()), dff.index)]

            n_gaps = len(dff.index)

            if n_gaps > 0:
                print "[ERROR] Found {} missing values !..".format(n_gaps)
                validity_message = 'Gap(s) detected'
                alert_level = 3
                date_fmt = "%Y-%m-%d"
                dff.index = dff.index.format(formatter=lambda x: parser.parse(str(x)).strftime(date_fmt))
                gaps_list = dff.index.values

    return validity_message, alert_level, gaps_list, n_gaps


def reorder_df(df):
    """
    Function reordering the DataFrame for better reading
    :param df: Input DataFrame (type: pandas DataFrame)
    :return:
        - df Reordered DataFrame (type: pandas DataFrame)
    """
    cols = ['filename', 'sql_name', 'readable', 'is_empty', 'category',
            'delimiter', 'ncols', 'has_date_col', 'today_date_included',
            'nrows_after_filter', 'frequency', 'benchmark_country', 'min_value', 'max_value',
            'max_abs_change', 'max_mac_values', 'max_cmv', 'max_ccv', 'ngaps', 'gap_dates',
            'fill_rate', 'integrity_check', 'validity_check', 'alert_level']
    return df[cols]


def validity_check(df, info_dict_df):
    """
    Function checking the validity of a DataFrame
    :param df: Input DataFrame (type: pandas DataFrame)
    :param info_dict_df: DataFrame dictionnary information (type: dict)
    :return:
        - DataFrame information dictionnary (type: dict)
    """
    vlist = []  # Validity messages list
    alist = []  # Alert level list

    # Checking min and max values
    val_min = info_dict_df['min_value']
    val_max = info_dict_df['max_value']
    validity_message, alert_level = check_min_max(df, val_min, val_max)
    vlist.append(validity_message)
    alist.append(alert_level)

    # Checking the maximum absolute change (mac)
    val_mac = info_dict_df['max_abs_change']
    validity_message, alert_level, max_over_val = check_mac(df, val_mac)
    info_dict_df['max_mac_values'] = max_over_val
    vlist.append(validity_message)
    alist.append(alert_level)

    # Checking maximum consecutive constant values (mccv)
    max_ccv = info_dict_df['max_ccv']
    validity_message, alert_level = check_mccv(df, max_ccv)
    vlist.append(validity_message)
    alist.append(alert_level)

    # Checking for gaps
    freq = info_dict_df['frequency']
    country = info_dict_df['benchmark_country']
    validity_message, alert_level, gaps_list, n_gaps = check_gaps(df, freq, country)
    info_dict_df['ngaps'] = n_gaps
    if n_gaps <= 10:
        info_dict_df['gap_dates'] = gaps_list
    vlist.append(validity_message)
    alist.append(alert_level)

    # Checking the fill rate
    validity_message, alert_level, ts_fill_rate = check_fill_rate(df, freq, thr_fr=0.90)
    info_dict_df['fill_rate'] = ts_fill_rate
    vlist.append(validity_message)
    alist.append(alert_level)

    # Alert Level
    alert_level = np.max(alist)
    info_dict_df['alert_level'] = alert_level
    # print 'alert_level = {}'.format(alert_level)

    # Validity Message
    if alert_level > 0:
        # print info_dict_df['sql_name']
        validity_message = ', '.join([vlist[i] for i, j in enumerate(vlist) if j != 'OK'])
        info_dict_df['validity_check'] = validity_message
        # print validity_message

    return info_dict_df


def check_date_conflicts(df1, df2):
    """
    Function checking date conflicts between two DataFrames
    :param df1: base directory 'windowed' DataFrame
    :param df2: latest data directory DataFrame
    :return:
        - is_conflict: indicates if there is any date conflicts (type: bool)
        - d: final dictionnary containing the information to be stored (type: dict)
    """
    is_conflict = False
    d = dict()  # Dictionary Initialization
    # Converting 'Date' column to 'DatetimeIndex' (for the diff operation below)
    df1.index = pd.DatetimeIndex(df1.index)
    df2.index = pd.DatetimeIndex(df2.index)
    col_name = df1.columns.values[0]
    # print col_name
    # Evaluating the differences between the DataFrames
    diff = df1.loc[df1[col_name] != df2[col_name]].index
    # Number of date conflicts
    nc = len(diff)
    if nc > 0:
        is_conflict = True
        print "[WARNING] Number of date conflicts = {}".format(nc)
        # Extracting the 'Base' sub-DataFrame containing the conflicts
        df1_conf = df1.loc[diff]
        # print "[INFO] 'Base' DataFrame dates conflicts :"
        # print df1_conf
        # Extracting the 'Latest data' sub-DataFrame containing the conflicts
        df2_conf = df2.loc[diff]
        # print "[INFO] 'Latest data' DataFrame dates conflicts :"
        # print df2_conf
        # Concatenating the two sub-DataFrames
        df_concat = pd.concat([df1_conf, df2_conf], axis=1)
        df_concat.columns = [col_name + '_old', col_name + '_new']
        # Building the final dict containing the information to be stored
        d = df_concat.T.to_dict('dict')
    return is_conflict, d


def write_list_to_csv(flist, fname):
    """
    Function writting accepted / denied files to csv files
    :param flist: list of files along with all their information (type: list)
    :param fname: type of file to be saved (accepted or denied) (type: string)
    :return:
    """
    lf = len(flist)
    if lf > 0:
        df = pd.DataFrame(flist)
        # Sorting rows according to the alert level, descending order
        df = df.sort(['alert_level', 'integrity_check', 'validity_check'], ascending=[0, 0, 0])
        # Reordering
        df = reorder_df(df)
        # Indexing
        df.set_index('filename', inplace=True)
        # Formatting the csv file to save
        fname = fname.replace(sep, ' ')
        # Saving the DataFrame in a csv file
        df.to_csv(fname, sep=',')
        # print 'Accepted Data List COMPLETE !'
        # print "Number of accepted files = {} / {} ({:.2f} %)".format(df_accepted.shape[0],len(activated_files), 100.0 * df_accepted.shape[0] / len(activated_files))


def data_control(data_dir, deriv_script, load_dfs):
    """
    Function performing the data control routine on a provided directory path
    :param data_dir: Directory to be data controlled (type: string)
    :param deriv_script: Derivation script name (type: string)
    :param load_dfs: Whether or not the derivation script is read (type: bool)
    :return:
    """
    # Initialization
    dfs = []
    list_accepted = []
    list_denied = []

    # Reading the Derivation Script
    if load_dfs:
        dfs, sheets = read_deriv_script(deriv_script)

    # Retrieving the list of files from the directory
    flist, nfiles = files_from_dir(data_dir)

    # for i, f in enumerate(flist[0:5]):  # TEST PURPOSE ONLY !!
    for i, f in enumerate(flist):
        print"[INFO] Processing file #{}: '{}'".format(i + 1, f)
        # Integrity Check
        df, df_info_dict = integrity_check(data_dir, f)

        # Filling the information dictionnary
        dict_values = fill_dict_from_df(df, dfs)
        df_info_dict['category'] = dict_values[0]
        df_info_dict['sql_name'] = dict_values[1]
        df_info_dict['frequency'] = dict_values[2]
        df_info_dict['benchmark_country'] = dict_values[3]
        df_info_dict['min_value'] = dict_values[4]
        df_info_dict['max_value'] = dict_values[5]
        df_info_dict['max_abs_change'] = dict_values[6]
        df_info_dict['max_cmv'] = dict_values[7]
        df_info_dict['max_ccv'] = dict_values[8]

        if df_info_dict['alert_level'] == 0:
            # Validity Check
            df_info_dict = validity_check(df, df_info_dict)

        if df_info_dict['alert_level'] == 0:
            # Accepted list
            list_accepted.append(df_info_dict)
        else:
            # Denied list
            list_denied.append(df_info_dict)

    # Writting output results to csv files
    write_list_to_csv(list_accepted, data_dir + '_accepted' + '.csv')
    write_list_to_csv(list_denied, data_dir + '_denied' + '.csv')

    print "\n[INFO] Data Control for '{}' Directory >> DONE !".format(data_dir)
