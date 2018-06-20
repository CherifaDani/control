# -*- coding: utf-8 -*-
# PydevCodeAnalysisIgnore
from __future__ import unicode_literals


"""
data_utils includes all functions that could be useful to the data management
"""

import os
import sys
import glob
import warnings
import csv
import numpy as np
import pandas as pd
from os.path import join, splitext, basename, dirname
import zipfile
import ast
from xlrd import XLRDError
import logging
from var_logger import setup_logging

# Launching logger
setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
logger.setLevel('DEBUG')


def find_delim(csv_path, delimiters=',;'):
    """
    Function used to find a delimiters in a csv file

    Parameters
    ----------
    csv_path : {string type} 
                Path to the csv file

    delimiters : {string type}
                 string with different possibles delimiters
                 ex: ',;' means that the function will test ',' and ';'

    Return
    ------
    dialect.delimiter : {char type}
                        the best delimiter of the csv among the given delimiters
    """
    # Test if the file exists
    assert os.path.isfile(csv_path), 'No csv file here %s' % csv_path
    f = open(csv_path, "rb")
    # Creation of a Sniffer object
    csv_sniffer = csv.Sniffer()
    # It reads nb_bytes bytes of the csv file and ...
    # ... chooses the best delimiter among the given delimiters
    try:
        dialect = csv_sniffer.sniff(f.readline(),
                                    delimiters=delimiters)
        f.close()
        return dialect.delimiter
    except csv.Error:
        f.close()
        return ';'


def make_index(csv_path, path_out=None, save=False, multi=False):
    """
    Function used to make an index for multi-asset learning
    from a unstack csv file

    Parameters
    ----------
    csv_path : {string type}
                  Path to the csv file

    path_out : {string type}, optional, default None
                  Path to write the file (used is save=True)

    save : {bool type}, optional, default false
           Option to save the index

    multi : {bool type}, optional, default false
            MultiIndex or not

    Return
    ------
    index : {DataFrame type} multiIndex time x stocks
        The multiIndex DataFrame for the learning
    """
    # Find the delimiter for the csv
    sep = find_delim(csv_path)
    # Loading the csv
    df = pd.read_csv(csv_path, sep=sep)
    # Drop the NaN line
    df.fillna(0, inplace=True)
    # First column (Date) in index / Stock in columns
    df.set_index(df.columns[0], inplace=True)

    if multi:
        df_stack = df.stack()
        # Get the multiIndex
        df_index = df_stack.index
        # Making of the index file
        index = pd.DataFrame()
        index['Dates'] = df_index.get_level_values(0)
        index['Ids'] = df_index.get_level_values(1)

    else:
        df_index = df.index
        index = pd.DataFrame()
        index['Dates'] = df_index

    index.set_index('Dates', inplace=True)

    if save:
        assert path_out is not None, \
            'You must give a path_out to write the index'
        # Saving of the index file
        # Name with the last date
        csv_name = 'Index ' + str(max(df.index)) + '.csv'
        path = os.path.join(path_out, csv_name)
        index.to_csv(path)

    return index


def find_bins(xcol, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    xcol : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                number of modalities

    Return
    ------
    bins : {ndarray type}
           the bins for disretization (result from numpy percentile function)
    """
    # Find the bins for nb_bucket
    q_list = np.arange(100.0/nb_bucket, 100.0, 100.0/nb_bucket)
    bins = np.array([np.nanpercentile(xcol, i) for i in q_list])

    if bins.min() != 0:
        test_bins = bins/bins.min()
    else:
        test_bins = bins

    # Test if we have same bins...
    while len(set(test_bins.round(5))) != len(bins):
        # Try to decrease the number of bucket to have unique bins
        nb_bucket -= 1
        q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
        bins = np.array([np.nanpercentile(xcol, i) for i in q_list])
        if bins.min() != 0:
            test_bins = bins/bins.min()
        else:
            test_bins = bins

    return bins


def discretize(xcol, nb_bucket, bins=None):
    """
    Function used to have discretize xcol in nb_bucket values
    if xcol is a real series and do nothing if xcol is a string series

    Parameters
    ----------
    xcol : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                number of modalities

    bins : {ndarray type}, optional, default None
           if you have already calculate the bins for xcol

    Return
    ------
    xcol_discretized : {Series type}
                       the discretization of xcol
    """

    if xcol.dtype.type != np.str_:
        # extraction of the list of xcol values
        notnan_vect = np.extract(np.isfinite(xcol), xcol)
        temp = pd.Series(np.nan, index=xcol.index)

        # Test if xcol have more than nb_bucket different values
        if len(set(notnan_vect)) > nb_bucket:
            if bins is None:
                bins = find_bins(xcol, nb_bucket)
            # discretization of the xcol with bins
            xcol_discretized = xcol.dropna().apply(lambda x:
                                                   np.digitize(x, bins=bins))
            temp.update(xcol_discretized)
            return temp
        
        # warnings.warn('Variable %s not discretize' %xcol.name)
        return xcol
    
    else:
        return xcol


def load_var(path, var_name, times_series=False):
    """
    Function used to load a variable from a path
    
    Parameters
    ----------
    path : {string type}
           Path to a csv or a directory

    var_name : {string type}
               Name of a column or a csv name

    times_series : {bool type} optional, default True,
                   Option to parse dates
    
    Return
    ------
    : {Series type}
    """
    # test if the path is to a file...
    if os.path.isfile(path):
        return load_from_file(path)
    
    # ... or to a directory
    elif os.path.isdir(path):
        return load_from_dir(path, var_name, times_series)
    
    else:
        warnings.warn('Path error: %s' % path)
        sys.exit()


def load_from_file(path):
    """
    Function used to load a variable from a csv

    Parameters
    ----------
    path : {string type}
           Path to a csv

    Return
    ------
    df: {Dataframe type}
         a dataframe with sorted index
    """

    sep = find_delim(path)
    extension = os.path.splitext(path)[1]

    # Test the type a file
    if extension == '.csv':
        df = pd.read_csv(path, sep=sep, index_col=0, parse_dates=True)
        nrows = df.shape[0]
        if not df.empty and nrows > 2:
            df.index = pd.DatetimeIndex(df.index)
            df.sort_index(ascending=True, inplace=True)
            df.columns = [x.upper() for x in df.columns]
            return df
        else:
            warnings.warn('Empty Dataframe: {} '.format(path))

    else:
        warnings.warn('Unknown extension: %s' % extension)
        sys.exit()


def load_from_dir(path, var_name, times_series):
    """
    Function used to load a variable from a csv
        
    Parameters
    ----------
    path : {string type}
           Path to a directory

    var_name : {string type}
               Name of a csv name

    times_series : {bool type}
                   Option to parse dates
    
    Return
    ------
    col: {Series type}
         the csv var_name stack to be a series
    """
    path = os.path.join(path, var_name)
    extension = os.path.splitext(path)[1]
    
    if extension != '.csv':
        path += '.csv'
    
    sep = find_delim(path)
    f = open(path)
    first_line = f.readline()
    splited_line = first_line.split(sep)
    f.close()
    
    if len(splited_line) > 3:
        col = pd.read_csv(path, sep=sep, parse_dates=times_series,
                          index_col=0)
        col = col.stack()
        col.name = var_name
    else:
        try:
            col = pd.read_csv(path, sep=sep, parse_dates=times_series,
                              index_col=[0, 1])
            col.sort_index(0, inplace=True)
            col = col[col.columns[0]]
            col.name = var_name
        except IndexError:
            col = pd.read_csv(path, sep=sep, parse_dates=times_series,
                              index_col=0)
            if type(col) == pd.DataFrame:
                col = col.iloc[:, 0]
                
            col.name = var_name
            
    return col


def load_index(path, times_series=True):
    """
    Function used to load a index 
        
    Parameters
    ----------
    path : {string type}
           Path to a csv

    times_series : {bool type}, optional, default True
                   Option to parse dates
    
    Return
    ------
    index_mat : {Series type}
                the index series for a multiIndex learning
    """
    sep = find_delim(path)
    index_mat = pd.read_csv(path, index_col=[0],
                            sep=sep, parse_dates=times_series)

    index_mat.index = index_mat.index.normalize()
    # To convert stock name in str
    if len(index_mat.columns) > 0:
        col_name = index_mat.columns[0]
        index_mat[col_name] = index_mat[col_name].astype('str')

        index_mat.set_index(keys=col_name, append=True, inplace=True)
        index_mat.sortlevel([0, 1], inplace=True)
    
    return index_mat


def apply_index(xcol, idx, idx_h=None, idx_m=None):
    """
    Function used to apply an multiIndex to xcol
        
    Parameters
    ----------
    xcol : {Series type} 
           xcol with the multiIndex

    idx : {Index type}
          the multiIndex
    
    Return
    ------
    indexed_col : {Series type} 
                  xcol with the given multiIndex
    """
    col_name = xcol.name
    indexed_col = pd.Series(np.nan, index=idx,
                            name=col_name,
                            dtype='float')

    # TODO replace update by xcol.reindex(idx, method='ffill')

    if idx_h is not None and idx_m is not None:
        is_multi_index = isinstance(xcol.index, pd.core.index.MultiIndex)
        if is_multi_index is False:
            if (all(xcol.index.hour > idx_h) or
                    all(xcol.index.hour == idx_h) and all(xcol.index.minute > idx_m)):
                xcol = xcol.shift()

            xcol.index = xcol.index.normalize()

        else:
            if (all(xcol.index.get_level_values(0).hour > 18) or
                    (all(xcol.index.get_level_values(0).hour == 18) and
                         all(xcol.index.get_level_values(0).minute > 30))):

                xcol = xcol.shift()

            xcol.index = xcol.index.get_level_values(0).normalize()

    indexed_col.update(xcol)

    return indexed_col  # xcol.reindex(idx, method='ffill')


def take_interval(xcol, dtend=None, dtstart=None, dayfirst=True):
    """
    Function used to take an interval of xcol
        
    Parameters
    ----------
    xcol : {Series type}

    dtend : {string type}, optional, default None
            lasted keeping date

    dtstart : {string type}, optional, default None
                first keeping date

    dayfirst : {bool type}, optional, default True

    Return
    ------
    xcol : {Series type}
           xcol between dtstart and dtend
    """
    is_multi_index = isinstance(xcol.index, pd.core.index.MultiIndex)
    if is_multi_index:
        first_index = xcol.index.get_level_values(0)
    else:
        first_index = xcol.index
    
    if len(xcol.index) == 0:
        warnings.warn('xcol is empty')
        return xcol
    
    if type(dtend) == int:
        if type(dtstart) == int:
            xcol = xcol.iloc[dtstart:dtend+1]
        else:
            xcol = xcol.iloc[: dtend+1]
    else:
        if dtstart is None or dtstart == '':
            dtstart = first_index[0]
        elif type(first_index[0]) == pd.tslib.Timestamp:
            dtstart = pd.to_datetime(dtstart, dayfirst=dayfirst)
        
        if dtend is None or dtend == '':
            dtend = first_index[-1]
        elif type(first_index[-1]) == pd.tslib.Timestamp:
            dtend = pd.to_datetime(dtend, dayfirst=dayfirst)
            
        if (dtstart <= first_index[0]) and (dtend >= first_index[-1]):
            warnings.warn('The chosen period is bigger or equal than the %s variable'
                          % xcol.name)
            return xcol
        elif dtend < dtstart:
            warnings.warn('Error with the date dstart=%s and dtend=%s' % (dtstart, dtend))
            sys.exit()
        else:
            if is_multi_index:
                xcol = xcol.loc[xcol.index.get_level_values(0) < dtend]
                xcol = xcol.loc[xcol.index.get_level_values(0) >= dtstart]
            else:
                xcol = xcol.loc[xcol.index < dtend]
                xcol = xcol.loc[xcol.index >= dtstart]
        
    return xcol


def get_variables(path, times_series=True):
    """
    Function used to get the all features
        
    Parameters
    ----------
    path : {string type}
           path to csv or a directory

    times_series : {bool type}, optional, default True

    Return
    ------
    variables : {list type} 
                The list of all features
    """
    if os.path.isfile(path):
        vars_list = get_variables_from_file(path, times_series)
    elif os.path.isdir(path):
        vars_list = get_variables_from_dir(path)
    else:
        warnings.warn('Error with the path %s' % path)
        return
    return vars_list


def get_variables_from_file(path, times_series):
    sep = find_delim(path)
    extension = os.path.splitext(path)[1]
    variables = []
    
    # Test the type a file
    if extension == '.csv':
        var_list = pd.read_csv(path, sep=sep, index_col=0).columns
    else:
        var_list = []

    if times_series:
        col_end = map(lambda i: 'M'+str(i), range(1, 11))
        variables = [col for col in var_list if col.split('_')[-1] in col_end
                     and '!' not in col]
        time_variables = [col for col in var_list if 'TIME' in col
                          and " " not in col]
        variables += time_variables
    
    return variables


def get_variables_from_dir(path):
    variables = glob.glob(path+'*.csv')
    variables = map(lambda f: f.split(r'/')[-1].split('\\')[-1].split('.csv')[0],
                    variables)
    return variables


def get_first_date(serie):
    is_multi_index = isinstance(serie.index, pd.core.index.MultiIndex)
    if is_multi_index:
        first_date = serie.dropna(inplace=False).index.get_level_values(0)[0]
    else:
        first_date = serie.dropna(inplace=False).index[0]
    
    return first_date


def get_nb_assets(serie):
    """
    Function used to get the number of stock
        
    Parameters
    ----------
    serie : {Series type}
            Serie with a multiIndex
    
    Return
    ------
    : {int type} 
      The number of stocks
    """
    if isinstance(serie.index, pd.core.index.MultiIndex) is False:
        return 1
    else:
        return int(len(set(serie.index.levels[1])))


def get_time_index(vect):
    """
    Function used to get the number of stock
        
    Parameters
    ----------
    vect : {Series type}
            Serie with a multiIndex
    
    Return
    ------
    time_index : {index type} 
                 The index of date for a multiIndex
    """
    if isinstance(vect.index, pd.core.index.MultiIndex):
        time_index = list(set(vect.index.levels[0]))
        time_index.sort()
    else:
        time_index = vect.index

    return time_index


def df_to_csv(df, csv_name):
    """
    Function used to save a dataframe to a csv file

    Parameters
    ----------
    df : {Pandas dataframe type}
                The dataframe to save

    csv_name : {string type}
                 The csv_name to save

    Return
    ------
    None
    """
    print 'Dataframe saved to csv'
    try:
        df.to_csv(csv_name, index=True, header=True)
        logger.info('Dataframe saved {}!'.format(csv_name))

    except TypeError:
        logger.exception('Failed to save the dataframe to the csv file, csv_name must be a string') 
    return None


def write_zip(path):
    """ write a zip file for the file indicated in the parameters

    Parameters
    ----------
    path : {String type}
               the full path  of the variable
    Return
    ------
    None
   """

    file_name = splitext(path)
    file_name = basename(file_name)
    zip_name = '{}{}'.format(file_name, '.zip')
    zf = zipfile.ZipFile(zip_name, mode='w')

    try:
        print 'adding {} to zip folder'.format(file_name)
        zf.write(path, basename(path))
        logger.info('')
    finally:
        print 'closing'
        zf.close()


def write_dict_to_csv(csv_name, f_dict, mode='w'):
    """
    Function used to save a dictionnary to a csv file

    Parameters
    ----------
    dict : {String type}
                The dictionnary to save

    csv_name : {string type}
                 The csv_name to save

    Return
    ------
    None
    """
    extension = os.path.splitext(csv_name)[1]
    # Test the type a file
    if extension != '.csv':
        csv_name = '{}{}'.format(csv_name, '.csv')

    try:
        with open(csv_name, mode) as csv_file:
            writer = csv.DictWriter(csv_file, f_dict.keys())
            writer.writeheader()
            writer.writerow(f_dict)
    except Exception as e:
        logger.exception('Failed to write dict to csv: {}'.format(csv_name))


def latestpath(path_base):
    """
    Function used to retrieve the path to latest data for a primary variable

    Parameters
    ----------
    path_base : {String type}
                The path to the base variable

    Return
    ------
    path_latest : {String type}
                  The full path to latest data of the same variable

    csv_name : {String type}
                The csv name of the csv path without extension
    """
    extension = os.path.splitext(path_base)[1]
    if extension != '.csv':
        warnings.warn('Unknown extension: {}'.format(extension))
    else:
        dir_path = dirname(path_base)
        basename_path = basename(path_base)
        lpath = dir_path.replace('Base', 'Latest data')
        path_latest = join(lpath, basename_path)

        # Retrieving the name of the csv_file without extension
        path_x = basename(path_base)
        csv_name, extension = os.path.splitext(path_x)
        csv_name = csv_name.upper()
        return path_latest, csv_name


def read_deriv_script(filename, sheet_name):
    """
    Function reading the derivation script from the filename
    and generating a DataFrame

    Parameters
    ----------
    filename : {String type}
                derivation script filename

    sheet_name : {string type}
                 The name of the excel sheet to load

    Return
    ------
    dfs : {Pandas dataframe}
            Dataframe of the read file
    """
    # Test the type of the file
    extension = os.path.splitext(filename)[1]
    if extension != '.xlsx':
        warnings.warn('Unknown extension: {}'.format(extension))

    # Attempt to extract the sheet name without loading the whole file
    xls = pd.ExcelFile(filename, on_demand=True)
    sheets = xls.sheet_names
    # Verifying if the sheet belongs to the derivation script
    if sheet_name in sheets:
        sheetname = sheet_name
    logger.info('Processing sheetname : {}'.format(sheetname))
    try:
        dfs = pd.read_excel(filename, sheetname=sheetname)

    except IOError:
        logger.exception('Impossible to read the derivation script !..')
    except XLRDError:
        logger.exception("No sheet named <'{}'> !".format(sheetname))

        # Replace all read NaN values by an empty string
        dfs.fillna('', inplace=True)
        dfs['SQL_Name'] = map(lambda x: x.upper(), dfs['SQL_Name'])
        logger.info('Derivation script {file} read !'.format(filename))
    return dfs


def test_cell(cell_name, cell_type, msg=''):
    if cell_type == 'str':
        if type(cell_name) not in [str]:
            logger.error('{} in {} is not a string'.format(msg, cell_name))
    if cell_type == 'int':
        if type(cell_name) not in [int]:
            logger.error('{} in {} is not an int'.format(msg, cell_name))
    if cell_type == 'float':
        if type(cell_name) not in [float]:
            logger.error('{} in {} is not a float'.format(msg, cell_name))


def fill_dict_from_df(dfs, variable_name):
    """
    Function generating a dictionnary when reading the dataframe
    for the derivation script

    Parameters
    ----------
    dfs : {dataframe type}
                derivation script dataframe

    variable_name : {string type}
                 The name of the variable to load

    Return
    ------
    var_dict : {dictionnary type}
            Dictionnary of the variable
    """
    if len(dfs) > 0:
        # Verifying the existence of the variable
        if variable_name not in dfs['SQL_Name'].values:
            raise Exception('Variable NOT found')

        # Retrieving row information from 'SQL_Name'
        sel_row = dfs.loc[dfs['SQL_Name'] == variable_name]
        logger.info('Retrieving informations for the variable: {}'.
                    format(variable_name))

        # Category
        cat_name = sel_row['Rubrique'].values
        cat_name = cat_name[0].encode('utf-8') if cat_name.size != 0 and not pd.isnull(cat_name)else ''
        test_cell(cat_name, 'str','Rubrique')

        # SQL Name
        sql_name = sel_row['SQL_Name'].values
        sql_name = sql_name[0].encode('utf-8') if sql_name.size != 0 else ''
        test_cell(sql_name, 'str', 'sql_name')
        
        # Frequency
        freq = sel_row['Frequence'].values
        freq = freq[0].encode('utf-8') if freq.size != 0 and not pd.isnull(freq) else ''
        test_cell(freq, 'str', 'Frequence')
        
        # Country
        country = sel_row['benchmark_country'].values
        country = country[0].encode('utf-8') if country.size != 0 and not pd.isnull(country) else ''
        test_cell(country, 'str', msg='benchmark_country')
        
        # Minimum Value
        min_val = sel_row['Minimum Value'].values
        min_val = np.float(min_val[0]) if min_val.size != 0 and not pd.isnull(min_val) else ''
        test_cell(min_val, 'float', 'Minimum Value')
        
        # Maximum Value
        max_val = sel_row['Maximum value'].values
        max_val = np.float(max_val[0]) if max_val.size != 0 and not pd.isnull(max_val) else ''
        test_cell(max_val, 'float', 'Maximum Value')


        # Maximum Absolute Change
        mac_val = sel_row['maximum absolute change'].values
        mac_val = np.float(mac_val[0]) if mac_val.size != 0 and not pd.isnull(mac_val) else ''
        test_cell(mac_val, 'float', 'maximum absolute change')

        # Maximum Consecutive Missing Values
        mcmv_val = sel_row['Max consecutive missing values'].values
        mcmv_val = np.int(mcmv_val[0]) if mcmv_val.size != 0 and not pd.isnull(mcmv_val) else ''
        test_cell(mcmv_val, 'int', 'Max consecutive missing values')

        # Maximum Consecutive Constant Values
        mccv_val = sel_row['Max consecutive constant values'].values
        mccv_val = np.int(mccv_val[0]) if mccv_val.size != 0 and not pd.isnull(mccv_val) else ''
        test_cell(mccv_val, 'int', 'Max consecutive constant values')

        # Path
        path = sel_row['Path'].values
        path = path[0].encode("utf-8") if path.size != 0 and not pd.isnull(path) else ''
        test_cell(path, 'str', 'Path')

        # Parents
        parents = sel_row['Parent List'].values
        parents = filter(lambda p: p == p, parents)

        if len(parents) > 0:
            parents = sel_row['Parent List'].values[0]
            parents = parents.replace('[', '[\'')
            parents = parents.replace(']', '\']')
            parents = parents.replace(',', '\',\'')
            parents = ast.literal_eval(parents)
        else:
            logger.debug('No parents')

        # Parameters
        parameters = sel_row['Parameters'].values
        parameters = filter(lambda p: p == p, parameters)

        if len(parameters) > 0:
            parameters = sel_row['Parameters'].values[0]
            parameters = ast.literal_eval(parameters)
            assert type(parameters) == dict, 'Parameters must be a dict'
        else:
            parameters = {}
            logger.debug('No Parameters!')

        dict_df = {'cat_name': cat_name,
                   'sql_name': sql_name,
                   'freq': freq,
                   'country': country,
                   'min_val': min_val,
                   'max_val': max_val,
                   'mac_val': mac_val,
                   'mcmv_val': mcmv_val,
                   'mccv_val': mccv_val,
                   'parents': parents,
                   'parameters': parameters,
                   'path': path
                   }
        return dict_df
    else:
        logger.error('Impossible to write the dictionnary of the variable!')