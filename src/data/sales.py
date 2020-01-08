#TODO: create a class "AggregateSales containing these function
#use the exemple in this link :  https://github.com/adeo/lmfr--bigdata--references-make-up/blob/master/src/data/ticket.py
import pandas as pd 
from .utils import read_sql
from skbox.connectors.bigquery import BigQuery
from skbox.connectors.teradata import Teradata
import os
#from functions import *
os.environ['TERADATA_UID'] = 'U_PROD_CHG_BIGDATA_001'
os.environ['TERADATA_PWD'] = '%BeegD@tq1'
PATH = ""
SOURCE_DICT = { 'teradata_sales':PATH+'load_sales.sql'}

class LoadSales():
    """
    """
    def __init__(self,data_source):
        self.data_source = data_source
        SALES = read_sql(SOURCE_DICT[self.data_source])
        SALES = SALES.replace('\n', '').replace('\r', '')
        teradata = Teradata()
        self.dataframe = teradata.select(SALES, chunksize = None)
    def agg_weekly_sales(self):
        """ aggregate daily sales to weekly sales per product
            Parameters
            ----------

            Return
            ------
        """
        return(self.dataframe.groupby(['NUM_ART', 'week_of_year','year_of_calendar']).agg({'QTE_VTE': 'sum'}).reset_index())
    def agg_monthly_sales(self):
        """ aggregate daily sales to weekly sales per product
            Parameters
            ----------
            Return
            ------
        """
        return(self.dataframe.groupby(['NUM_ART', 'month_of_year','year_of_calendar']).agg({'QTE_VTE': 'sum'}).reset_index())


