import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot  as plt
from data import LoadSales
import logging as log
from models import *
if __name__ == "__main__":
    ############################################
    # >> Process start
    log.info('** Load Data...')
    SALES_df = LoadSales('teradata_sales')
    list_ref = np.unique(SALES_df.dataframe.NUM_ART)
    RESULT_FORECAST = pd.DataFrame()
    #TODO : parallelize the outer loop is better than inner loop?
    for ref in list_ref :
        print("processing the SKU n° " + str (ref))
        #TODO:add weekly process
        df_i = SALES_df.transform(freq ="month",sku = ref)
        data = df_i.sales
        # data split
        n_test = 14 
        step = 12
        if sum(data.values[:-n_test]) == 0 :
            print("not enough data to forecast the sku n° " +str(ref))
        else :    
            # model configs
            #TODO:adding/deleting other elemnts in seasonal
            cfg_list = exp_smoothing_configs(seasonal=[6,12])
            scores = grid_search(data.values, cfg_list, n_test)
            print('***********simulation done**********')
            #selecting the best model (a low number of RMSE)
            best_cfg = eval( scores.pop(0)[0] )
            y_hat_test = forecast_model(data.values[:-n_test], config = best_cfg, h = n_test - 1 )
            #forecast the data
            y_hat = forecast_model(data.values, config = best_cfg, h = step - 1 )
            #crete the dates for the forecasts
            date_projection = pd.date_range(df_i.date[len(df_i)-1],periods = step+1 ,freq='MS')[1:]
            data_final = df_i.copy()
            data_final["forecast"] = np.NaN
            data_final.forecast[-n_test:]= y_hat_test
            #adding the forecast to the past data
            data_future = pd.DataFrame()
            data_future["date"] = date_projection
            data_future["sales"] = np.NaN
            data_future["forecast"] = y_hat
            data_final = data_final.append(data_future)
            data_final = data_final.reset_index(drop = True)
            data_final["sku"] = ref
            #data_final.to_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ str(ref)+".csv")
            plt.figure(figsize=(10,6))
            plt.plot(data_final.date,data_final.sales)
            plt.plot(data_final.date,data_final.forecast)
            plt.savefig("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/plots/"+ str(ref) + '.png')
            RESULT_FORECAST = RESULT_FORECAST.append(data_final)
            #TODO: si forecast est negative => rendre la valeur = 0
            #TODO: ajouter le coeff de variation pour chaque produit 
    RESULT_FORECAST.to_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ "RES_FINAL" +".csv")
