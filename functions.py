def agg_weekly_sales(table_daily):
    """ aggregate daily sales to weekly sales per product
        Parameters
        ----------
            table_daily : table containing daily_sales
        Return
        ------
            table_weekly : pandas dataframe
    """
    table_weekly = table_daily.groupby(['NUM_ART', 'week_of_year','year_of_calendar']).agg({'QTE_VTE': 'sum'}).reset_index()
    return(table_weekly)


def agg_monthly_sales(table_daily):
    """ aggregate daily sales to weekly sales per product
        Parameters
        ----------
            table_daily : table containing daily_sales
        Return
        ------
            table_monthly : pandas dataframe
    """
    table_monthly = table_daily.groupby(['NUM_ART', 'month_of_year','year_of_calendar']).agg({'QTE_VTE': 'sum'}).reset_index()
    return(table_monthly)