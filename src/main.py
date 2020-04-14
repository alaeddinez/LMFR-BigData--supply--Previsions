import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot  as plt
from data import LoadSales
import logging as log
from models import *
from interpretability import evaluate_all,diag_ability_performance
from tqdm import tqdm
import requests


#TODO:add the pid trick to see if program has finished executing or not
if __name__ == "__main__":
    ############################################
    # >> Process start
    log.info('** Load Data...')
    web_hook_url = "https://hooks.slack.com/services/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" 
    # SALES_df = LoadSales('Gardena', 'teradata')
    SALES_df = LoadSales('Gardena_bq', 'bq')
    list_ref = np.unique(SALES_df.dataframe.NUM_ART)
    RESULT_FORECAST = pd.DataFrame()
    # TODO: add tqdm on the loop
    # TODO : parallelize the outer loop is better than inner loop?
    for ref in tqdm(list_ref):
        print("processing the SKU n° "+str (ref))
        # TODO:add weekly process
        df_i = SALES_df.transform(freq="month", sku=ref)
        data = df_i.sales
        data = df_i.sales.astype(float)
        # data split
        n_test = 14
        step = 12
        if sum(data.values[:-n_test]) == 0 :
            print("not enough data to forecast the sku n° " +str(ref))
        else :    
            # model configs
            #TODO:adding/deleting other elemnts in seasonal
            cfg_list = exp_smoothing_configs(seasonal=[6, 12])
            scores = grid_search(data.values, cfg_list, n_test)
            print('***********simulation done**********')
            #selecting the best model (a low number of RMSE)
            best_cfg = eval(scores.pop(0)[0])
            y_hat_test = forecast_model(data.values[:-n_test], config = best_cfg, h = n_test - 1 )
            errors = data.tail(n_test).values - y_hat_test
            errors_std = np.std(errors)
            #forecast the data
            y_hat = forecast_model(data.values, config = best_cfg, h = step - 1 )
            upper, lower = conf_int(errors_std, y_hat, 1.150)
            #crete the dates for the forecasts
            date_projection = pd.date_range(df_i.date[len(df_i)-1],periods = step+1 ,freq='MS')[1:]
            data_final = df_i.copy()
            data_final["forecast"] = np.NaN
            data_final.forecast[-n_test:] = y_hat_test
            data_final["upper"] = np.NaN
            data_final["lower"] = np.NaN
            #adding the forecast to the past data
            data_future = pd.DataFrame()
            data_future["date"] = date_projection
            data_future["sales"] = np.NaN
            data_future["forecast"] = y_hat
            data_future["upper"] = upper
            data_future["lower"] = lower
            data_final = data_final.append(data_future)
            data_final = data_final.reset_index(drop=True)
            data_final["sku"] = ref
            #data_final.to_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ str(ref)+".csv")
            #TODO : ameliorer le plot : ajouter 3 couleur (train, test ,futur)
            # plt.figure(figsize=(10,6))
            # plt.plot(data_final.date,data_final.sales)
            # plt.plot(data_final.date,data_final.forecast)
            # plt.savefig("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/plots/"+ str(ref) + '.png')
            RESULT_FORECAST = RESULT_FORECAST.append(data_final)
            
            #TODO : ajouter l'intervalle de confiance !!
            #TODO: ajouter le coeff de variation pour chaque produit 
    #writing the result in a csv
    #TODO: si forecast est negative => rendre la valeur = 0
    RESULT_FORECAST.forecast[ RESULT_FORECAST.forecast < 0 ] = 0
    RESULT_FORECAST.to_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ "RES_FINAL" +".csv")
    print("saving forecasts")
    slack_message = {'text' : 'saving forecasts in csv ! :)'}
    requests.post(url=web_hook_url,
                  data=json.dumps(slack_message))   


    #using the interpretability model 
    data = pd.read_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ "RES_FINAL" +".csv")
    #using only the test set part
    data = data[[ 'date', 'sales', 'forecast','sku']].dropna()
    DAP = diag_ability_performance(data.sales.values,data.forecast.values, data.date.values, data.sku.values)
    DAP.to_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ "ROC" +".csv",sep= ";",index =False) 
    #this csv is used in a powerbi dashboard to visualize which sku have better performed and which ones created created a huge gap 
    list_dicts = []
    for sku in np.unique(data.sku) :     
        data_sku = data[data.sku == sku ]
        kpi_sku = {"sku" : sku}
        kpi_sku.update(evaluate_all(data_sku.sales.values,data_sku.forecast.values))
        list_dicts.append(kpi_sku)
    #create the dataframe of the kpi for each sku from the dict created
    kpi_df = pd.DataFrame(list_dicts,columns = list_dicts[0].keys())
    kpi_df.to_csv("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/tables/"+ "kpi_df" +".csv",sep= ";",index =False)