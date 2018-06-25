# coding: utf-8

from __future__ import division


import data_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import basename, splitext


path = 'GOV_JPN_1Y_Z250D.csv'
var_name = splitext(path)[0]



# read csv file
df = data_utils.load_var(path, var_name)
df.plot()
plt.show()


def zero_cross(arr):
    neg_pos = ((arr[:-1] * arr[1:]) < 0).sum()
    # zcr = (1/T)*sum((s(t)*(st-1) < 0))
    zcr = neg_pos / len(arr)
    return zcr


# print zero_cross()
print zero_cross(df.values)
