#%%
import datetime as dt
import pandas as pd
from arch import arch_model

import warnings 
warnings.filterwarnings('ignore')

from arch.unitroot import ADF

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera

f_path = r"C:\Users\krkoo\OneDrive\Desktop\ESSEX Univ_studies\Dissertation meetings\meeting 1\realized_ret_variance.csv"
# f_path = r"m:\pc\desktop\Dissertation meetings\meeting 1\realized_ret_variance.csv"

df = pd.read_csv(f_path, header=0)


# st = dt.datetime(1988, 1, 1)
# en = dt.datetime(2018, 1, 1)


df['Time'] = pd.to_datetime(df['Time'].astype(str), format='%Y%m%d')
df.set_index('Time', inplace=True)

df_data={}
index_list=[]
for i in range(0, len(df.columns), 3):
    # print(i)
    c_name=df.columns[i][:10]
    index_list.append(c_name)
    # j=i%3
    df_data[c_name] = df.iloc[:,i:i+2]
    # df_data[c_name].ffill(inplace=True)
    df_data[c_name].dropna(inplace=True)
    df_data[c_name].iloc[:,0] = df_data[c_name].iloc[:,0]
    df_data[c_name].iloc[:,1] = df_data[c_name].iloc[:,1]


returns = df_data[index_list[0]].iloc[:,0]
variance_series = df_data[index_list[0]].iloc[:,1]

# returns = 100 * market.pct_change().dropna()
ax = returns.plot()
xlim = ax.set_xlim(returns.index.min(), returns.index.max())

# ## ADF test
# adf = ADF(returns)
# print(adf.summary().as_text())

#%%    

'''

Model Types -- 
Type 1 : GARCH(1,1) model
Type 2 : GJR-GARCH(1,1) model
Type 3 : TARCH(1,1) model
Type 4 : E-GARCH(1,1) model

Diistribution Types -- 
Type A : Normal dist
Type B : Students-T dist

'''


model_type = ['1' , '2' , '3' , '4' ]
distr_type = ['B']

# model_type = ['2' ]
# distr_type = ['A', 'B' ]


for mt in model_type:
    for dt in distr_type:
        if mt=='1' and dt=='A':
            am_garch = arch_model(returns, vol='Garch', p=1, q=1 , dist="Normal") # add for t-distribution  dist="StudentsT"
        if mt=='1' and dt=='B':
            am_garch = arch_model(returns, vol='Garch', p=1, q=1 , dist="StudentsT") # add for t-distribution  dist="StudentsT"

        if mt=='2' and dt=='A':
            am_garch = arch_model(returns, p=1, o=1, q=1 , dist="Normal") # add for t-distribution  dist="StudentsT"
        if mt=='2' and dt=='B':
            am_garch = arch_model(returns, p=1, o=1, q=1 , dist="StudentsT") # add for t-distribution  dist="StudentsT"

        if mt=='3' and dt=='A':
            am_garch = arch_model(returns, p=1, o=1, q=1, power=1.0 , dist="Normal") # add for t-distribution  dist="StudentsT"
        if mt=='3' and dt=='B':
            am_garch = arch_model(returns, p=1, o=1, q=1, power=1.0 , dist="StudentsT") # add for t-distribution  dist="StudentsT"

        if mt=='4' and dt=='A':
            am_garch = arch_model(returns, vol='EGarch', p=1, q=1 , dist="Normal") # add for t-distribution  dist="StudentsT"
        if mt=='4' and dt=='B':
            am_garch = arch_model(returns, vol='EGarch', p=1, q=1 , dist="StudentsT") # add for t-distribution  dist="StudentsT"

    
        res_garch = am_garch.fit(update_freq=2)
        print(res_garch.summary())
        
        
        # Plotting the realized variance vs. conditional volatility
        plt.figure(figsize=(12, 6))
        plt.plot(variance_series, label='Realized Variance', color='blue')
        plt.plot(res_garch.conditional_volatility**2, label='Conditional Variance', color='red')
        plt.title('Realized Variance vs. Conditional Variance {}'.format('res_garch'))
        plt.legend()
        
        # Standardized residuals
        standardized_residuals = res_garch.resid / res_garch.conditional_volatility
        
        # 1. ACF of Residuals
        plt.figure(figsize=(10, 6))
        plot_acf(standardized_residuals, lags=20)
        plt.title("ACF of Standardized Residuals")
        plt.show()
        
        # 2. ACF of Squared Residuals
        plt.figure(figsize=(10, 6))
        plot_acf(standardized_residuals**2, lags=20)
        plt.title("ACF of Squared Standardized Residuals")
        plt.show()
        
        # 3. Ljung-Box Test on Residuals
        ljung_box_results = acorr_ljungbox(standardized_residuals, lags=[10, 20], return_df=True)
        print("Ljung-Box Test on Residuals:")
        print(ljung_box_results)
        
        # 4. Ljung-Box Test on Squared Residuals
        ljung_box_results_sq = acorr_ljungbox(standardized_residuals**2, lags=[10, 20], return_df=True)
        print("Ljung-Box Test on Squared Residuals:")
        print(ljung_box_results_sq)
        
        # 5. Jarque-Bera Test for Normality
        jb_test = jarque_bera(standardized_residuals)
        print(f"Jarque-Bera Test Statistic: {jb_test[0]}, P-Value: {jb_test[1]}")
        
        # 6. Plot of Standardized Residuals
        plt.figure(figsize=(10, 6))
        plt.plot(standardized_residuals)
        plt.title("Standardized Residuals")
        plt.show()


#%%

# ## GARCH(1,1) MODEL -- CONSTANT MEAN
# am_garch = arch_model(returns, vol='Garch', p=1, q=1 , dist="Normal") # add for t-distribution  dist="StudentsT"
# res_garch = am_garch.fit(update_freq=2)
# print(res_garch.summary())


# ##  OTHER FIGURES  ## 
# # Assuming 'res' is the result from your GARCH model fitting
# # Plotting the returns vs. conditional volatility
# plt.figure(figsize=(12, 6))
# plt.plot(returns, label='Returns', color='blue')
# plt.plot(res_garch.conditional_volatility, label='Conditional Volatility', color='red')
# plt.title('Returns vs. Conditional Volatility {}'.format('res_garch'))
# plt.legend()
# plt.show()

# # Plotting the conditional variance
# plt.figure(figsize=(12, 6))
# plt.plot(res_garch.conditional_volatility**2, color='green')
# plt.title('Conditional Variance {}'.format('res_garch'))
# plt.show()

# # Plotting the residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_garch.resid, color='purple')
# plt.title('Residuals {}'.format('res_garch'))
# plt.show()

# # Plotting the standardized residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_garch.resid / res_garch.conditional_volatility, color='orange')
# plt.title('Standardized Residuals {}'.format('res_garch'))
# plt.show()

# # Plotting the ACF of standardized residuals
# plt.figure(figsize=(12, 6))
# plot_acf(res_garch.resid / res_garch.conditional_volatility, lags=40)
# plt.title('ACF of Standardized Residuals {}'.format('res_garch'))
# plt.show()


# ########

# ## GJR-GARCH MODEL
# am_gjr = arch_model(returns, p=1, o=1, q=1)
# res_gjr = am_gjr.fit(update_freq=5, disp="off")
# print(res_gjr.summary())

# # Assuming 'res' is the result from your GARCH model fitting
# # Plotting the returns vs. conditional volatility
# plt.figure(figsize=(12, 6))
# plt.plot(returns, label='Returns', color='blue')
# plt.plot(res_gjr.conditional_volatility, label='Conditional Volatility', color='red')
# plt.title('Returns vs. Conditional Volatility {}'.format('res_gjr'))
# plt.legend()
# plt.show()

# # Plotting the conditional variance
# plt.figure(figsize=(12, 6))
# plt.plot(res_gjr.conditional_volatility**2, color='green')
# plt.title('Conditional Variance {}'.format('res_gjr'))
# plt.show()

# # Plotting the residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_gjr.resid, color='purple')
# plt.title('Residuals {}'.format('res_gjr'))
# plt.show()

# # Plotting the standardized residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_gjr.resid / res_gjr.conditional_volatility, color='orange')
# plt.title('Standardized Residuals {}'.format('res_gjr'))
# plt.show()

# # Plotting the ACF of standardized residuals
# plt.figure(figsize=(12, 6))
# plot_acf(res_gjr.resid / res_gjr.conditional_volatility, lags=40)
# plt.title('ACF of Standardized Residuals {}'.format('res_gjr'))
# plt.show()


# #########

# ## TARCH MODEL
# am_tarch = arch_model(returns, p=1, o=1, q=1, power=1.0)
# res_tarch = am_tarch.fit(update_freq=5, disp="off")
# print(res_tarch.summary())

# # Assuming 'res' is the result from your GARCH model fitting
# # Plotting the returns vs. conditional volatility
# plt.figure(figsize=(12, 6))
# plt.plot(returns, label='Returns', color='blue')
# plt.plot(res_tarch.conditional_volatility, label='Conditional Volatility', color='red')
# plt.title('Returns vs. Conditional Volatility {}'.format('res_tarch'))
# plt.legend()
# plt.show()

# # Plotting the conditional variance
# plt.figure(figsize=(12, 6))
# plt.plot(res_tarch.conditional_volatility**2, color='green')
# plt.title('Conditional Variance {}'.format('res_tarch'))
# plt.show()

# # Plotting the residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_tarch.resid, color='purple')
# plt.title('Residuals {}'.format('res_tarch'))
# plt.show()

# # Plotting the standardized residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_tarch.resid / res_tarch.conditional_volatility, color='orange')
# plt.title('Standardized Residuals {}'.format('res_tarch'))
# plt.show()

# # Plotting the ACF of standardized residuals
# plt.figure(figsize=(12, 6))
# plot_acf(res_tarch.resid / res_tarch.conditional_volatility, lags=40)
# plt.title('ACF of Standardized Residuals {}'.format('res_tarch'))
# plt.show()



# #########

# ## E-GARCH MODEL
# am_egarch = arch_model(returns, vol='EGarch', p=1, q=1)
# res_egarch = am_egarch.fit(update_freq=5, disp="off")
# print(res_egarch.summary())

# # Assuming 'res' is the result from your GARCH model fitting
# # Plotting the returns vs. conditional volatility
# plt.figure(figsize=(12, 6))
# plt.plot(returns, label='Returns', color='blue')
# plt.plot(res_egarch.conditional_volatility, label='Conditional Volatility', color='red')
# plt.title('Returns vs. Conditional Volatility {}'.format('res_egarch'))
# plt.legend()
# plt.show()

# # Plotting the conditional variance
# plt.figure(figsize=(12, 6))
# plt.plot(res_egarch.conditional_volatility**2, color='green')
# plt.title('Conditional Variance {}'.format('res_egarch'))
# plt.show()

# # Plotting the residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_egarch.resid, color='purple')
# plt.title('Residuals {}'.format('res_egarch'))
# plt.show()

# # Plotting the standardized residuals
# plt.figure(figsize=(12, 6))
# plt.plot(res_egarch.resid / res_egarch.conditional_volatility, color='orange')
# plt.title('Standardized Residuals {}'.format('res_egarch'))
# plt.show()

# # Plotting the ACF of standardized residuals
# plt.figure(figsize=(12, 6))
# plot_acf(res_egarch.resid / res_egarch.conditional_volatility, lags=40)
# plt.title('ACF of Standardized Residuals {}'.format('res_egarch'))
# plt.show()



