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
import os
import json
from skbox.connectors.bigquery import BigQuery
from google.cloud import bigquery
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/alaeddinez/bigdata_bigquery_service_account_key_dev.json'
#TODO:add the pid trick to see if program has finished executing or not

############################################
# >> Process start
log.info('** Load Data...')
web_hook_url = "https://hooks.slack.com/services/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" 
# SALES_df = LoadSales('Gardena', 'teradata')
#SALES_df = LoadSales('gardena_daily', 'bq')
SALES_df = LoadSales('pv', 'bq')
df_i = SALES_df.dataframe
df_i = df_i.tail(60)
data = df_i.PV
data = df_i.PV.astype(float)
# TODO : delete last day bcz it is not done yet 
# data split
n_test = 4
step = 7
# model configs
#TODO:adding/deleting other elemnts in seasonal
cfg_list = exp_smoothing_configs(seasonal=[7,30])
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
#date_projection = pd.date_range(df_i.date[len(df_i)-1],periods = step+1 ,freq='MS')[1:]
date_projection = pd.date_range( df_i.date.tail(1).astype("str").values[0],periods = step+1 ,freq='D')[1:]

data_final = df_i.copy()
data_final["forecast"] = np.NaN
data_final.forecast[-n_test:] = y_hat_test
data_final["upper"] = np.NaN
data_final["lower"] = np.NaN
#adding the forecast to the past data
data_future = pd.DataFrame()
data_future["date"] = date_projection
data_future["PV"] = np.NaN
data_future["forecast"] = y_hat
data_future["upper"] = upper
data_future["lower"] = lower
data_final = data_final.append(data_future)
data_final = data_final.reset_index(drop=True)
#TODO : ameliorer le plot : ajouter 3 couleur (train, test ,futur)
# plt.figure(figsize=(10,6))
# plt.plot(data_final.date,data_final.sales)
# plt.plot(data_final.date,data_final.forecast)
# plt.savefig("/home/alaeddinez/MyProjects/LMFR-BigData--supply--Previsions/output/plots/"+ str(ref) + '.png')
#writing the result in a csv
#TODO: si forecast est negative => rendre la valeur = 0
data_final.forecast[ data_final.forecast < 0 ] = 0
# delete hh:mm:ss part from the date
data_final['date'] = data_final['date'].astype("str").str.split(' ').str[0]
  
#using the interpretability model 
data = data_final.copy()
data = data[['date', 'PV', 'forecast']].dropna()
list_dicts = []
kpi_sku = {"indicators" : "values"}
kpi_sku.update(evaluate_all(data.PV.values,data.forecast.values))
list_dicts.append(kpi_sku)
#create the dataframe of the kpi for each sku from the dict created
kpi_df = pd.DataFrame(list_dicts,columns = list_dicts[0].keys())
kpi_df[['mse','mape','smape','mae','mda']]

data_final['mape'] = kpi_df.mape

SCHEMA_PREV = [
  {
    "description": "date",
    "mode": "REQUIRED",
    "name": "date",
    "type": "STRING"
  },
  {
    "description": "nombre de pagevues reel",
    "mode": "NULLABLE",
    "name": "PV",
    "type": "FLOAT"
  },
  {
    "description": "nombre de pagevues predit",
    "mode": "NULLABLE",
    "name": "forecast",
    "type": "FLOAT"
  }
    ,
  {
    "description": "borne sup de l IC",
    "mode": "NULLABLE",
    "name": "upper",
    "type": "FLOAT"
  },

  {
    "description": "borne inf de l IC",
    "mode": "NULLABLE",
    "name": "lower",
    "type": "FLOAT"
  }
]

OutputProjectBQ = "big-data-dev-lmfr"
OutputDataSetBQ = "supply_test"
OutputTableBQ = "pv_covid_prev"
load_config = bigquery.LoadJobConfig()
load_config.write_disposition = "WRITE_TRUNCATE"
load_config.schema = SCHEMA_PREV
client = bigquery.Client(OutputProjectBQ)
dataset_ref = client.dataset(OutputDataSetBQ)
table_ref = dataset_ref.table(OutputTableBQ)
client.load_table_from_dataframe(data_final,table_ref,job_config=load_config).result()
# slack access bot token
web_hook_url = "https://hooks.slack.com/services/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" 

slack_message = {'text' : 'PREV  PV UPLOADED! :)'}
requests.post(url=web_hook_url,
                data=json.dumps(slack_message))  

