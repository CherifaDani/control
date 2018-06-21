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
 
 
df = pd.read_csv('18 06 Derived/Y/FUT_SP500_RDTR20.csv', index_col=0, parse_dates=True)
#===============================================================================
# nan_values = df[df.columns[0]].notnull().astype(int).cumsum()
# x = df[df.columns[0]].isnull().astype(int).groupby(nan_values).sum()
# print x.values
# print np.max(x.values)
#===============================================================================

#df.fillna('xxx')


def nan_df(df, thr):
    alert_level = 0
    c_message = 'OK'
    nan_values = df[df.columns[0]].notnull().astype(int)
    nan_values = nan_values.cumsum()
    nan_count = df[df.columns[0]].isnull().astype(int)
    nan_count = nan_count.groupby(nan_values).sum()
    consecutive_nans = np.max(nan_count.values)
    if consecutive_nans > thr:
        alert_level = 2
        c_message = 'Many consecutive Nans'
    return c_message, alert_level, consecutive_nans
print nan_df(df, 5)