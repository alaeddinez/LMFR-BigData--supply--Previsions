
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def diag_ability_performance(
                y_true_id, y_prev_id, filter_col,
                id_col):
    
    """ this function allows to visualize by a filter_col a graphical evaluation of the forecast.
        the graph shows in x-axis the cumulative percentage of the absolute values
        percentage errors and in the y-axis the percent cumulative sales (up to 100% of total sales).
        the interpretation is then:
        -the more the curve is on the left , the better the quality of the forecast
        
        For a better interpretation, the output can be used in a powerbi dashboard for a Dynamic visualisation  
            
    
        Parameters
        -----------
            y_true_id : list 
                True value per ID
           
            y_prev_id : list 
                predicted value per ID
            
            filter_col : list 
                True value per ID
                
            id_col : list 
                True value per ID
            
        Return
        ------
            DAP: pandas dataframe
                
        
        Example
        -------
            .. code-block:: python
                
                from skbox.interpretability import diag_ability_performance
                
                #An exemple of the Input : 
                #      date  actual  prev       art  
                #0  11-2018     0.0   0.0  36633331    
                #1  12-2018     0.0   0.0  36633331    
                #2   1-2019     0.0   0.0  36633331    
                #3   2-2019     0.0   0.0  36633331    
                #4   3-2019     0.0   0.0  36633331    
                
                #Create the data and visualize the result by the ID
                DAP = diag_ability_performance(ROC.actual.values, ROC.prev.values, ROC.date.values, ROC.art.values)
    """
    #create a dataframe from the inputs which is lists 
    DAP = pd.DataFrame({'y_true_id':y_true_id, 'y_prev_id':y_prev_id, 'filter_col':filter_col, 'id_col':id_col})
    #create a variable which contains the residuals
    DAP['error'] = DAP.y_true_id - DAP.y_prev_id
    
    #transfrom all the residuals to absolute values 
    DAP['error_abs'] = DAP.error.abs()
    
  
    sum_error = DAP[['filter_col','error_abs']].groupby(['filter_col']).sum().reset_index()
    sum_error = sum_error.rename(columns={'error_abs':'sum_error'})

    sum_target = DAP[['filter_col','y_true_id']].groupby(['filter_col']).sum().reset_index()
    sum_target = sum_target.rename(columns={'y_true_id':'sum_target'})


 
    #join the dataframe rows  with the variables containing the sum by using the filter_col
    DAP = pd.merge(DAP,sum_target,on=['filter_col'],how='left')
    DAP = pd.merge(DAP,sum_error,on=['filter_col'],how='left')
    
    #calculate the proportions 
    DAP["prop.y_true_id"] = DAP["y_true_id"]/DAP["sum_target"]
    DAP["prop.error"] = (DAP["error"]/DAP["sum_error"]).abs()
    
    #to sort values by descending 
    DAP["actual_neg"] = -DAP["y_true_id"]
    DAP = DAP.sort_values(['filter_col','actual_neg'])
    
    #calculate cumulative sums 
    Sa = DAP[["filter_col","prop.y_true_id"]] \
                .groupby(['filter_col'])['prop.y_true_id'] \
                .apply(lambda x: x.cumsum()) 

    Er = DAP[["filter_col","prop.error"]] \
                .groupby(['filter_col'])['prop.error'] \
                .apply(lambda x: x.cumsum()) 
    
    #renaming the columns
    Sa = pd.DataFrame(Sa).rename(columns = {'prop.y_true_id':'Sa'})
    Er = pd.DataFrame(Er).rename(columns = {'prop.error':'Er'})
    
    #adding the cumulative sums to the initial dataframe
    DAP = pd.concat([DAP, Sa, Er], axis=1)
    
    # #plot to visualise the cumulative sum on y-axis versus the cumulative error on x-axis for each filter_col
    # filter_list = np.unique(DAP.filter_col.astype('str').values)
    # row = int(len(filter_list)/3)+1
    # c,l = 0,0
    
    # fig, ax = plt.subplots(row,3,figsize = (18,16))
    # for f in filter_list:
    #     ax_ = ax[l,c]
    #     c += 1
    #     if c == 3:
    #         c = 0
    #         l += 1
    #     tmp = DAP[DAP.filter_col.isin([f])].copy()
    #     ax_.plot(tmp.Er, tmp.Sa, 'g+', label=f)
    #     ax_.set_xlabel('Cumulative absolute error')
    #     ax_.set_ylabel('Cumulative TARGET')
    #     ax_.set_title('Diagnostic ability of prediction (DAP)')
    #     ax_.grid(True)
    #     ax_.legend()
    return DAP


