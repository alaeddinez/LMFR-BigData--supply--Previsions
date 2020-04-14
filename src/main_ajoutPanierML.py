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

K = 67
############################################
# >> Process start
log.info('** Load Data...')
web_hook_url = "https://hooks.slack.com/services/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" 

df_ajout_panier = LoadSales('forecast_ajout_panier', 'bq').dataframe
#df_ajout_panier = LoadSales('no_oreo_ajout', 'bq').dataframe

df_page_views = LoadSales('forecast_pv', 'bq').dataframe

df_ap = df_ajout_panier.copy()
df_pv = df_page_views.copy()

df_ap.nb_ajout_panier.fillna(df_ap.forecast, inplace=True)
df_pv.PV.fillna(df_pv.forecast, inplace=True)

result_forecast = pd.merge(df_ap[["date", "nb_ajout_panier"]],
                 df_pv[["date", "PV"]],
                 on='date', 
                 how='left')

result_forecast["reel_ap"] = result_forecast.nb_ajout_panier

result_forecast = result_forecast.tail(K).reset_index()


#source de donnees
#utiliser les PV (real)
#utiliser les ap (real en lag + forecast Ã  date et lag) ? a voir
#ajoutable au panier ( mediane,moyenne ,quantile ..) (oreo : apres 1 avril : source / avant : une autre source de donnees)

#dummies (pas d'appels aux source de donnees)
#avant et apres confinement
#depollution ? entre le 15 et le 20  ..[zone chelou]
#algo start ( elastic search, score 3 conf ,reco filtre .. voir post renault)

#ajouter les données de nombre de possibilité d'ajout au panier
ap_avail = LoadSales('panier_avail', 'bq').dataframe
ap_avail['mean_ap_avail'] = ap_avail.loc[:, ap_avail.columns != 'date'].mean(axis=1)
ap_avail = ap_avail[["date","COVER_20P_NB_ART","COVER_50P_NB_ART","COVER_80P_NB_ART","mean_ap_avail"]]
ap_avail.date = ap_avail.date.astype("str") 
result_forecast.date = result_forecast.date.astype("str") 

result_forecast_1 = pd.merge(result_forecast,
                 ap_avail,
                 on='date', 
                 how='left')


#remplacer les panier_avail pour les donnees avant 1  avril
result_forecast_1.loc[result_forecast.tail(7).index, 'reel_ap'] = np.nan
result_forecast_1.loc[result_forecast.date[result_forecast.date < '2020-04-01'].index,'COVER_20P_NB_ART'] = result_forecast_1.COVER_20P_NB_ART[result_forecast.date ==  '2020-04-01'].values[0]
result_forecast_1.loc[result_forecast.date[result_forecast.date < '2020-04-01'].index,'COVER_50P_NB_ART'] = result_forecast_1.COVER_50P_NB_ART[result_forecast.date ==  '2020-04-01'].values[0]
result_forecast_1.loc[result_forecast.date[result_forecast.date < '2020-04-01'].index,'COVER_80P_NB_ART'] = result_forecast_1.COVER_80P_NB_ART[result_forecast.date ==  '2020-04-01'].values[0]
result_forecast_1.loc[result_forecast.date[result_forecast.date < '2020-04-01'].index,'mean_ap_avail'] = result_forecast_1.mean_ap_avail[result_forecast.date ==  '2020-04-01'].values[0]

#pour fill les panier_avail dans le futur avec la derniere vues !
result_forecast_1 = result_forecast_1.fillna(method='ffill')

#ajout variable Renault
result_forecast_1["search"] = 0
result_forecast_1["score3confinement"]  = 0
result_forecast_1["ranking"] = 0
result_forecast_1["reco"] = 0

result_forecast_1.search[result_forecast_1.date >= "2020-04-01"] = 1
result_forecast_1.score3confinement[result_forecast_1.date >= "2020-04-07"] = 1
result_forecast_1.ranking[result_forecast_1.date >= "2020-04-06"] = 1 
result_forecast_1.reco[result_forecast_1.date >= "2020-04-03"] = 1 

# IMPORTANT  TO MANAGE Y VARIABLE  : reel_ap
result_forecast_1.loc[result_forecast_1.tail(7).index, 'reel_ap'] = np.nan

result_forecast_1["weird_period"] = 0 
result_forecast_1.weird_period[ (result_forecast_1.date >= "2020-03-15") & (result_forecast_1.date <= "2020-03-22")  ] = 1 


lags_np = pd.concat([ result_forecast_1.nb_ajout_panier.shift(), result_forecast_1.nb_ajout_panier.shift(2),
          result_forecast_1.nb_ajout_panier.shift(3),result_forecast_1.nb_ajout_panier.shift(7),
          result_forecast_1.nb_ajout_panier.shift(10),result_forecast_1.nb_ajout_panier.shift(15),
          result_forecast_1.nb_ajout_panier.shift(30)
            ], axis=1)
lags_np.columns = ["lag1_np","lag2_np","lag3_np","lag7_np","lag10_np","lag15_np","lag30_np"]
lags_np= pd.DataFrame(lags_np)

lags_pv = pd.concat([ result_forecast_1.PV.shift(), result_forecast_1.PV.shift(2),
                    result_forecast_1.PV.shift(3),result_forecast_1.PV.shift(7),
            ], axis=1)
lags_pv.columns = ["lag1_pv","lag2_pv","lag3_pv","lag7_pv"]
lags_pv= pd.DataFrame(lags_pv)


result_forecast_1 = pd.DataFrame(result_forecast_1)
final_data = pd.concat([result_forecast_1,lags_np,lags_pv],axis = 1)


#fill the na made with lag
final_data = final_data.fillna(method='bfill')

#change date and extract
final_data['date'] = pd.to_datetime(final_data['date'])
final_data['week'] = final_data['date'].dt.week
final_data['day_week'] = final_data['date'].dt.dayofweek
features_days = pd.get_dummies( final_data['date'].dt.dayofweek.astype("str"))
final_data = pd.concat([final_data,features_days ],axis = 1)


final_data_forecast = final_data.tail(7)
final_data_train_test = final_data.head(K - 7 )

####################################################################################
#
#                               MACHINE LEARNING PART
####################################################################################
# Labels are the values we want to predict
labels = np.array(final_data_train_test['reel_ap'])
# Remove the labels from the features
features = final_data_train_test.copy()
features  = features.drop('reel_ap', axis = 1)
features  = features.drop('date', axis = 1)
features  = features.drop('nb_ajout_panier', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10, random_state = 42)


from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators =1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
MAPE = np.mean(mape)
MAPE

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

######################################################################
#                   NEW RANDOM FOREST
########################################################################

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 500, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('lag1_np'), 
                     feature_list.index('lag10_np'),
                     #feature_list.index('index'),
                     feature_list.index('PV'),
                     feature_list.index('lag2_np'),
                     #feature_list.index('lag7_np'),
                     #feature_list.index('lag15_np'),
                     feature_list.index('lag1_pv'),
                     feature_list.index('mean_ap_avail'),
                     feature_list.index('COVER_20P_NB_ART'),
                     feature_list.index('COVER_50P_NB_ART'),
                     feature_list.index('COVER_80P_NB_ART'),
                     feature_list.index('ranking'),
                     #feature_list.index('lag30_np'),
                     #feature_list.index('lag3_np'),
                     feature_list.index('lag3_pv'),
                     feature_list.index('week'),

]




train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train 
from sklearn.linear_model import ARDRegression, LinearRegression
clf_train = ARDRegression(compute_score=True)
clf_train.fit(train_important, train_labels)

predictions = clf_train.predict(test_important)

errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
MAPE = np.mean(mape)
MAPE

###################" use the same variables for the new algorithm containing all (train and test = present !!)"###################
present_important = final_data_train_test.copy()
present_important  = present_important.drop('reel_ap', axis = 1)
present_important  = present_important.drop('date', axis = 1)
present_important  = present_important.drop('nb_ajout_panier', axis = 1)


futur_important =  final_data_forecast.copy()
futur_important  = futur_important.drop('reel_ap', axis = 1)
futur_important  = futur_important.drop('date', axis = 1)
futur_important  = futur_important.drop('nb_ajout_panier', axis = 1)

present_important = np.array(present_important)[:, important_indices]
futur_important = np.array(futur_important)[:, important_indices]


############################grid #############

# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores

# # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(present_important, np.array(final_data['reel_ap'].head(K-7)))
# rf_random.best_params_

#60
#rf_grid = RandomForestRegressor(n_estimators = 400, min_samples_split= 5, min_samples_leaf =  1, max_features= 'auto', max_depth= 90, bootstrap =  False)

#30
#rf_grid = RandomForestRegressor(n_estimators = 1000, min_samples_split= 5, min_samples_leaf =  4, max_features= 'auto', max_depth=30, bootstrap =  False)

# rf_grid.fit(present_important, np.array(final_data['reel_ap'].head(K-7)))

# predictions = rf_grid.predict(futur_important)
# predictions

# final_data["prev_ap"] = np.nan
# final_data.prev_ap.tail(7).values[0:7] = predictions


# df_forecast = final_data[["date","nb_ajout_panier","PV","reel_ap","prev_ap"]]


# df_forecast


# from sklearn.linear_model import LinearRegression
# reg = LinearRegression().fit(present_important, np.array(final_data['reel_ap'].head(K-7)))
# reg.predict(futur_important)


# Fit the ARD Regression
from sklearn.linear_model import ARDRegression, LinearRegression
clf = ARDRegression(compute_score=True)
clf.fit(present_important, np.array(final_data['reel_ap'].head(K-7)))

predictions = clf.predict(futur_important)

final_data['prev_ap'] = np.nan
final_data.prev_ap.tail(7).values[0:7] = predictions

df_forecast = final_data[["date","reel_ap","prev_ap"]]
df_forecast.date =  df_forecast.date.astype(str)


SCHEMA_PREV = [
  {
    "description": "date",
    "mode": "REQUIRED",
    "name": "date",
    "type": "STRING"
  },
  {
    "description": "nombre ajout panier reel",
    "mode": "NULLABLE",
    "name": "reel_ap",
    "type": "FLOAT"
  },
  {
    "description": "nombre ajout panier predit",
    "mode": "NULLABLE",
    "name": "prev_ap",
    "type": "FLOAT"
  }
    
]

OutputProjectBQ = "big-data-dev-lmfr"
OutputDataSetBQ = "supply_test"
OutputTableBQ = "nb_ajout_panier_prev_ml"
load_config = bigquery.LoadJobConfig()
load_config.write_disposition = "WRITE_TRUNCATE"
load_config.schema = SCHEMA_PREV
client = bigquery.Client(OutputProjectBQ)
dataset_ref = client.dataset(OutputDataSetBQ)
table_ref = dataset_ref.table(OutputTableBQ)
client.load_table_from_dataframe(df_forecast,table_ref,job_config=load_config).result()
# slack access bot token
web_hook_url = "https://hooks.slack.com/services/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" 

slack_message = {'text' : 'PREV  nb_ajout_panier_prev_ml LEARNING UPLOADED! :)'}
requests.post(url=web_hook_url,
                data=json.dumps(slack_message))  


