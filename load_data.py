from skbox.connectors.bigquery import BigQuery
from skbox.connectors.teradata import Teradata
import pandas as pd 
import os
from utils import read_sql
from functions import *

os.environ['TERADATA_UID'] = 'U_PROD_CHG_BIGDATA_001'
os.environ['TERADATA_PWD'] = '%BeegD@tq1'
PATH = ''
sales = read_sql(PATH+'load_sales.sql')
sales = sales.replace('\n', '').replace('\r', '')
teradata = Teradata()

#loading original data (daily sales)
four_sales = teradata.select(sales, chunksize = None)

#transforming the daily sales to weekly sales
weekly_sales = agg_weekly_sales(four_sales)
weekly_sales.to_csv("weekly_sales.csv", sep=";",index = False)

#transforming the daily sales to monthly sales
monthly_sales = agg_monthly_sales(four_sales)
monthly_sales.to_csv("monthly_sales.csv", sep=";",index = False)

