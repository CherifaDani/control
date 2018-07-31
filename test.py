import pandas as pd
import numpy as np

#===============================================================================
# df = pd.DataFrame({"Person":
#                     ["John", "Myla", None, np.nan, np.nan, 'edee', 'fdfs', None, None,
#                      None, None]}
#                     )
# mis_val_percent = df.isnull().sum() / len(df)
# 
# #print df.isnull().sum() / len(df)
# 
# 
# x = df.Person.isnull().astype(int).groupby(df.Person.notnull().astype(int).cumsum()).sum()
# print x.values
# print np.max(x.values)
#  
#  
#===============================================================================
 
 
df = pd.read_csv('test/MACRO_INVEST_CONFID_USA_NEUT_LAST.csv', index_col=0, parse_dates=True)


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


print(infer_freq(df))








#===============================================================================
# nan_values = df[df.columns[0]].notnull().astype(int).cumsum()
# x = df[df.columns[0]].isnull().astype(int).groupby(nan_values).sum()
# print x.values
# print np.max(x.values)
#===============================================================================

#df.fillna('xxx')

#
# def nan_df(df, thr):
#     alert_level = 0
#     c_message = 'OK'
#     nan_values = df[df.columns[0]].notnull().astype(int)
#     nan_values = nan_values.cumsum()
#     nan_count = df[df.columns[0]].isnull().astype(int)
#     nan_count = nan_count.groupby(nan_values).sum()
#     consecutive_nans = np.max(nan_count.values)
#     if consecutive_nans > thr:
#         alert_level = 2
#         c_message = 'Many consecutive Nans'
#     return c_message, alert_level, consecutive_nans
# print nan_df(df, 5)