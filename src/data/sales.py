import pandas as pd 
from .utils import read_sql
from skbox.connectors.bigquery import BigQuery
from skbox.connectors.teradata import Teradata
import os
import sys
import time
import logging as log
os.environ['TERADATA_UID'] = 'U_PROD_CHG_BIGDATA_001'
os.environ['TERADATA_PWD'] = '%BeegD@tq1'
#PATH = ""
PATH = './data/'

SOURCE_DICT = { 'teradata_sales': PATH+'load_sales.sql',
                'louis_sales' : PATH + 'load_sales_test.sql',
                'louis_sales_bq': PATH + 'load_sales_bq.sql',
                'ga_pageviews': PATH+'load_monthly_pageviews.sql',
                'test' : PATH + 'bq_test.sql'
               }


class LoadSales():
    """
    """
    def __init__(self,data_source, option_source):
        if option_source == "teradata":
            self.data_source = data_source
            SALES = read_sql(SOURCE_DICT[self.data_source])
            SALES = SALES.replace('\n', '').replace('\r', '')
            teradata = Teradata()
            self.dataframe = teradata.select(SALES, chunksize = None)
        elif option_source == "bq":
            self.data_source = data_source
            sql_req = read_sql(SOURCE_DICT[self.data_source])
            sql_req = sql_req.replace('\n', ' ').replace('\r', ' ')
            bq = BigQuery()
            df_BQ = bq.select(sql_req)
            self.dataframe = df_BQ
        else :
            print("wrong option")
    def agg_weekly_sales(self):
        """ aggregate daily sales to weekly sales per product
            Parameters
            ----------

            Return
            ------
        """
        return(self.dataframe.groupby(['NUM_ART', 'week_of_year','year_of_calendar']).agg({'QTE_VTE': 'sum'}).reset_index())
    def agg_monthly_sales(self):
        """ aggregate daily sales to monthly sales per product
            Parameters
            ----------
            Return
            ------
        """
        return(self.dataframe.groupby(['NUM_ART', 'month_of_year','year_of_calendar']).agg({'QTE_VTE': 'sum'}).reset_index())
    def transform(self,sku,freq,del_current=True):
        """ transform the agg data to the suitable format to make forecasts
            Parameters
            ----------
            Return
            ------
        """
        if freq == 'month':
            #TODO:the renaming is not general
            monthly_sales = self.agg_monthly_sales()
            monthly_sales = monthly_sales.rename(
            columns={"year_of_calendar": "year",
            "month_of_year": "month",
            "QTE_VTE" : "sales"}
            )
            #creating a column date with date format to make forecasts
            monthly_sales["day"] = 1
            monthly_sales["date"] = pd.to_datetime(
            monthly_sales[["year", "month", "day"]])
            #create a commun date range for all the units (in our case :productsku)
            minDate = min(monthly_sales.date)
            maxDate = max(monthly_sales.date) 
            date_range = pd.date_range(minDate, maxDate, 
            freq=pd.DateOffset(months = 1, day = 1))
            date_range_df = pd.DataFrame(date_range, columns=["date"])
            #deleting the date of the current month (we suppose that the data for the current month is not complete)
            if del_current :
                date_range_df.drop(date_range_df.tail(1).index,inplace=True)
            #selecting the productsku from the parameter passed in the method
            monthly_sales = monthly_sales[monthly_sales.NUM_ART == sku]
            #left join with all the dates in date_range_df
            monthly_sales = pd.merge(
            date_range_df, 
            monthly_sales[["date" ,"sales"]],
            on =["date"], how = "left")
            transformed_sales = monthly_sales.fillna(0)
        #TODO : prepare the transformation in case of week
        elif freq == 'week':
            weekly_sales = self.agg_weekly_sales()
        else:
            log.info("ERROR argument perimeter")
            sys.exit()
        

        return(transformed_sales)


        

class LoadPageviews():
    """

    """
    def __init__(self, data_source):
        self.data_source = data_source
        sql_req = read_sql(SOURCE_DICT[self.data_source])
        sql_req = sql_req.replace('\n', ' ').replace('\r', ' ')
        bq = BigQuery()
        df_BQ = bq.select(sql_req)
        self.dataframe = df_BQ

    def return_table(self):
        """ Parameters
            ----------

            Return
            ------
        """
        return(self.dataframe)