from __future__ import division
import numpy as np
import data_utils
import pandas as pd
from dateutil import parser
#import control_var as cv
from matplotlib import pyplot
import control_utils
# Librairies

import logging
from var_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
logger.setLevel('DEBUG')


# print df
# df.plot()
# pyplot.show()

df = data_utils.load_var('STR_USD_3M_DACE_1_20_100.csv', 'GOV_JPN_1Y_Z250D')


