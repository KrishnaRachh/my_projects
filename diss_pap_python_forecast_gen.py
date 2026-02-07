import datetime as dt
import sys
import numpy as np
import pandas as pd
from arch import arch_model

import warnings 
warnings.filterwarnings('ignore')

from arch.unitroot import ADF

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error



f_path = r"C:\Users\krkoo\OneDrive\Desktop\ESSEX Univ_studies\Dissertation meetings\meeting 1\realized_ret_variance.csv"
# f_path = r"m:\pc\desktop\Dissertation meetings\meeting 1\realized_ret_variance.csv"

df = pd.read_csv(f_path, header=0)

test_split = "1-1-2005"
test_split_date = dt.datetime(2005, 1, 1)

# st = dt.datetime(1988, 1, 1)
# en = dt.datetime(2018, 1, 1)


df['Time'] = pd.to_datetime(df['Time'].astype(str), format='%Y%m%d')
df.set_index('Time', inplace=True)

df_data={}
index_list=[]
for i in range(0, len(df.columns), 3):
    print(i)
    c_name=df.columns[i][:10]
    index_list.append(c_name)
    # j=i%3
    df_data[c_name] = df.iloc[:,i:i+2]
    # df_data[c_name].ffill(inplace=True)
    df_data[c_name].dropna(inplace=True)
    df_data[c_name].iloc[:,0] = df_data[c_name].iloc[:,0]
    df_data[c_name].iloc[:,1] = df_data[c_name].iloc[:,1]

mse_df = pd.DataFrame(columns=index_list)
for ind in range(len(index_list)):
    try:
        print(index_list[ind])
        returns = df_data[index_list[ind]].iloc[:,0]
        realized_variance = df_data[index_list[ind]].iloc[:,1]
        # realized_variance_train = realized_variance.loc[:test_split_date]
        
        realized_variance_train = realized_variance.copy()
        realized_variance_train.loc[realized_variance.index > test_split_date] = np.nan
        realized_variance_test = realized_variance.loc[test_split_date:]
    
        # ax = returns.plot()
        # xlim = ax.set_xlim(returns.index.min(), returns.index.max())
        # ax = realized_variance.plot()
        
        # # ADF test
        # adf = ADF(returns)

        
       
        #%%
        #### FORECASTING GARCH(1,1) MODEL -- CONSTANT MEAN 
        am_garch = arch_model(returns, vol='Garch', p=1, q=1 , dist="StudentsT") # add for t-distribution  dist="StudentsT"
        res_garch = am_garch.fit(update_freq=2)
        print(res_garch.summary())
        
        ## Fixed window forecasting -- Enter the last date and model will fix that date and make forecasts ahead
        sim_forecasts = res_garch.forecast(start="1-1-2005", method="simulation", horizon=1)
        # print(sim_forecasts.variance.dropna().head())
        
        fcst_name = sim_forecasts.variance.columns
        # plt.plot(sim_forecasts.variance.dropna())
        # plt.legend(fcst_name)
        
        # print(" Fixed Length MSE is : ", mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna()))
        mse_df.loc['FL_Garch', index_list[ind]] = mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna())
        plt.plot(realized_variance_test - sim_forecasts.variance.dropna()['h.1'])
        
        # ## Rolling window forecasting -- given a window n , model will consider prev n observation to forecast ahead and it will roll forward as forecast is computed
        # n=2000
        # index = returns.index
        # start_loc = 0
        # end_loc = np.where(index >= "1-1-2005")[0].min()
        # forecasts = {}
        # i=end_loc
        # for i in range(end_loc, len(index)):
        #     sys.stdout.write(".")
        #     sys.stdout.flush()
        #     res_garch = am_garch.fit(first_obs=i-n, last_obs=i-1, disp="off")
        #     temp = res_garch.forecast(start = i, method="simulation" , horizon=1).variance
        #     fcast = temp.iloc[i]
        #     # print(fcast)
        #     forecasts[fcast.name] = fcast
        # print()
        # # print(pd.DataFrame(forecasts).T)
        
        # forecast_df = (pd.DataFrame(forecasts)).T
        # # print(forecast_df)
        
        # print(" Rolling window MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        
        
        ## Recursive Forecasting -- similar to rolling except first observation doesn't change (aka Expanding window)
        index = returns.index
        start_loc = 0
        end_loc = np.where(index >= "1-1-2005")[0].min()
        forecasts = {}
        for i in range(len(index) - end_loc):
            sys.stdout.write(".")
            sys.stdout.flush()
            res_garch = am_garch.fit(last_obs=i + end_loc-1, disp="off")
            temp = res_garch.forecast(method="simulation"  , horizon=1).variance
            fcast = temp.iloc[i+ end_loc]
            # print(fcast)
            forecasts[fcast.name] = fcast
        print()
        # print(pd.DataFrame(forecasts).T)
        forecast_df = (pd.DataFrame(forecasts)).T
        
        # print(" Recursive MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        mse_df.loc['Recc_Garch', index_list[ind]] = mean_squared_error(realized_variance_test , forecast_df.dropna())
        
        plt.plot(realized_variance_test - forecast_df.dropna()['h.1'])
        
        
        #%%
        #### FORECASTING GJR-GARCH(1,1) MODEL -- CONSTANT MEAN 
        am_garch = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist="StudentsT") # add for t-distribution  dist="StudentsT"
        res_garch = am_garch.fit(update_freq=2)
        print(res_garch.summary())
        
        ## Fixed window forecasting -- Enter the last date and model will fix that date and make forecasts ahead
        sim_forecasts = res_garch.forecast(start="1-1-2005", method="simulation", horizon=1)
        # print(sim_forecasts.variance.dropna().head())
        
        fcst_name = sim_forecasts.variance.columns
        # plt.plot(sim_forecasts.variance.dropna())
        # plt.legend(fcst_name)
        
        # print(" Fixed Length MSE is : ", mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna()))
        mse_df.loc['FL_gjrgarch', index_list[ind]] = mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna())
    
        # ## Rolling window forecasting -- given a window n , model will consider prev n observation to forecast ahead and it will roll forward as forecast is computed
        # n=1000
        # index = returns.index
        # start_loc = 0
        # end_loc = np.where(index >= "1-1-2005")[0].min()
        # forecasts = {}
        # for i in range(end_loc, len(index)):
        #     sys.stdout.write(".")
        #     sys.stdout.flush()
        #     res_garch = am_garch.fit(first_obs=i-n, last_obs=i-1, disp="off")
        #     temp = res_garch.forecast(start = i, method="simulation" , horizon=1).variance
        #     fcast = temp.iloc[i]
        #     # print(fcast)
        #     forecasts[fcast.name] = fcast
        # print()
        # # print(pd.DataFrame(forecasts).T)
        
        # forecast_df = (pd.DataFrame(forecasts)).T
        # # print(forecast_df)
        
        # print(" Rolling window MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        
        
        ## Recursive Forecasting -- similar to rolling except first observation doesn't change (aka Expanding window)
        index = returns.index
        start_loc = 0
        end_loc = np.where(index >= "1-1-2005")[0].min()
        forecasts = {}
        for i in range(len(index) - end_loc):
            sys.stdout.write(".")
            sys.stdout.flush()
            res_garch = am_garch.fit(last_obs=i + end_loc-1, disp="off")
            temp = res_garch.forecast(method="simulation"  , horizon=1).variance
            fcast = temp.iloc[i+ end_loc]
            # print(fcast)
            forecasts[fcast.name] = fcast
        print()
        # print(pd.DataFrame(forecasts).T)
        forecast_df = (pd.DataFrame(forecasts)).T
        
        # print(" Recursive MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        mse_df.loc['Recc_gjrgarch', index_list[ind]] = mean_squared_error(realized_variance_test , forecast_df.dropna())
    
        
        
        #%%
        #### FORECASTING TARCH(1,1) MODEL -- CONSTANT MEAN 
        am_garch = arch_model(returns, vol='Garch', p=1, o=1, q=1, power=1 , dist="StudentsT") # add for t-distribution  dist="StudentsT"
        res_garch = am_garch.fit(update_freq=2)
        print(res_garch.summary())
        
        ## Fixed window forecasting -- Enter the last date and model will fix that date and make forecasts ahead
        sim_forecasts = res_garch.forecast(start="1-1-2005", method="simulation", horizon=1)
        # print(sim_forecasts.variance.dropna().head())
        
        fcst_name = sim_forecasts.variance.columns
        # plt.plot(sim_forecasts.variance.dropna())
        # plt.legend(fcst_name)
        
        # print(" Fixed Length MSE is : ", mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna()))
        mse_df.loc['FL_tarch', index_list[ind]] = mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna())
    
        # ## Rolling window forecasting -- given a window n , model will consider prev n observation to forecast ahead and it will roll forward as forecast is computed
        # n=1000
        # index = returns.index
        # start_loc = 0
        # end_loc = np.where(index >= "1-1-2005")[0].min()
        # forecasts = {}
        # for i in range(end_loc, len(index)):
        #     sys.stdout.write(".")
        #     sys.stdout.flush()
        #     res_garch = am_garch.fit(first_obs=i-n, last_obs=i-1, disp="off")
        #     temp = res_garch.forecast(start = i, method="simulation" , horizon=1).variance
        #     fcast = temp.iloc[i]
        #     # print(fcast)
        #     forecasts[fcast.name] = fcast
        # print()
        # # print(pd.DataFrame(forecasts).T)
        
        # forecast_df = (pd.DataFrame(forecasts)).T
        # # print(forecast_df)
        
        # print(" Rolling window MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        
        
        ## Recursive Forecasting -- similar to rolling except first observation doesn't change (aka Expanding window)
        index = returns.index
        start_loc = 0
        end_loc = np.where(index >= "1-1-2005")[0].min()
        forecasts = {}
        for i in range(len(index) - end_loc):
            sys.stdout.write(".")
            sys.stdout.flush()
            res_garch = am_garch.fit(last_obs=i + end_loc-1, disp="off")
            temp = res_garch.forecast(method="simulation"  , horizon=1).variance
            fcast = temp.iloc[i+ end_loc]
            # print(fcast)
            forecasts[fcast.name] = fcast
        print()
        # print(pd.DataFrame(forecasts).T)
        forecast_df = (pd.DataFrame(forecasts)).T
        
        # print(" Recursive MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        mse_df.loc['Recc_tarch', index_list[ind]] = mean_squared_error(realized_variance_test , forecast_df.dropna())
    
        
        #%%
        #### FORECASTING EGARCH(1,1) MODEL -- CONSTANT MEAN 
        am_garch = arch_model(returns, vol='EGarch', p=1, q=1, dist="StudentsT") # add for t-distribution  dist="StudentsT"
        res_garch = am_garch.fit(update_freq=2)
        print(res_garch.summary())
        
        ## Fixed window forecasting -- Enter the last date and model will fix that date and make forecasts ahead
        sim_forecasts = res_garch.forecast(start="1-1-2005", method="simulation", horizon=1)
        # print(sim_forecasts.variance.dropna().head())
        
        fcst_name = sim_forecasts.variance.columns
        # plt.plot(sim_forecasts.variance.dropna())
        # plt.legend(fcst_name)
        
        # print(" Fixed Length MSE is : ", mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna()))
        mse_df.loc['FL_egarch', index_list[ind]] = mean_squared_error(realized_variance_test , sim_forecasts.variance.dropna())
    
        # ## Rolling window forecasting -- given a window n , model will consider prev n observation to forecast ahead and it will roll forward as forecast is computed
        # n=1000
        # index = returns.index
        # start_loc = 0
        # end_loc = np.where(index >= "1-1-2005")[0].min()
        # forecasts = {}
        # for i in range(end_loc, len(index)):
        #     sys.stdout.write(".")
        #     sys.stdout.flush()
        #     res_garch = am_garch.fit(first_obs=i-n-1, last_obs=i-1, disp="off")
        #     temp = res_garch.forecast(start = i, method="simulation" , horizon=1).variance
        #     fcast = temp.iloc[i]
        #     # print(fcast)
        #     forecasts[fcast.name] = fcast
        # print()
        # # print(pd.DataFrame(forecasts).T)
        
        # forecast_df = (pd.DataFrame(forecasts)).T
        # # print(forecast_df)
        
        # print(" Rolling window MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        
        
        ## Recursive Forecasting -- similar to rolling except first observation doesn't change (aka Expanding window)
        index = returns.index
        start_loc = 0
        end_loc = np.where(index >= "1-1-2005")[0].min()
        forecasts = {}
        for i in range(len(index) - end_loc):
            sys.stdout.write(".")
            sys.stdout.flush()
            res_garch = am_garch.fit(last_obs=i + end_loc-1, disp="off")
            temp = res_garch.forecast(method="simulation"  , horizon=1).variance
            fcast = temp.iloc[i+ end_loc]
            # print(fcast)
            forecasts[fcast.name] = fcast
        print()
        # print(pd.DataFrame(forecasts).T)
        forecast_df = (pd.DataFrame(forecasts)).T
        
        # print(" Recursive MSE is : ", mean_squared_error(realized_variance_test , forecast_df.dropna()))
        mse_df.loc['Recc_egarch', index_list[ind]] = mean_squared_error(realized_variance_test , forecast_df.dropna())
    except:
        pass
