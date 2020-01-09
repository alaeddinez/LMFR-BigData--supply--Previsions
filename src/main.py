import pandas as pd
import numpy as np
import datetime
from data import LoadSales
import logging as log
from models import classic_ts
if __name__ == "__main__":

    ############################################
    # >> Process start
    log.info('** Load Data...')
    data = LoadSales('teradata_sales')
    df_i = data.transform(freq ="month",sku = 69239730)
    data = df_i.sales
    # data split
    n_test = 12
    # model configs
    cfg_list = exp_smoothing_configs(seasonal=[12])
