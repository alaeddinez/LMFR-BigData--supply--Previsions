import pandas as pd
import numpy as np
import datetime
from data import LoadSales
import logging as log
from models import *
if __name__ == "__main__":

    ############################################
    # >> Process start
    log.info('** Load Data...')
    data = LoadSales('teradata_sales')
    df_i = data.transform(freq ="month",sku = 69239730)
    data = df_i.sales
    # data split
    n_test = 14
    # model configs
    #TODO:adding/deleting other elemnts in seasonal
    cfg_list = exp_smoothing_configs(seasonal=[0,6,12])
    scores = grid_search(data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
    

    # grid search
    scores = grid_search(np.array(data), cfg_list, n_test)

    #forecast the data