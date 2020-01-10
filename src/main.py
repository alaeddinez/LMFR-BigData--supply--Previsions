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
    #TODO:add weekly process
    df_i = data.transform(freq ="month",sku = 69239730)
    data = df_i.sales
    # data split
    n_test = 14 
    step = 12
    # model configs
    #TODO:adding/deleting other elemnts in seasonal
    cfg_list = exp_smoothing_configs(seasonal=[6,12])
    scores = grid_search(data.values, cfg_list, n_test)
    print('simulation done')
    #selecting the best model (a low number of RMSE)
    best_cfg = eval( scores.pop(0)[0] )
    #forecast the data
    y_hat = forecast_model(data.values, config = best_cfg, h = step - 1 )
    #crete the dates for the forecasts
    date_projection = pd.date_range(df_i.date[len(df_i)-1],periods = step+1 ,freq='MS')[1:]
    #adding the forecast to the past data
    data_future = pd.DataFrame()
    data_future["date"] = date_projection
    data_future["sales"] = y_hat
    data_final = df_i.copy()
    data_final["type_value"] ="a"
    data_future["type_value"]= "f"
    data_final = data_final.append(data_future)
    data_final = data_final.reset_index(drop = True)

    #get the predictions on the test set to visually compare too