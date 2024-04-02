#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split


# In[55]:


df = pd.read_csv(r"C:\Users\eoinh\OneDrive\Documents\Master_Excel.csv")


# In[56]:


df['Log_CPI_Home'] = np.log(df["CPI_Home"])


# In[57]:


df['Log_CPI_Foreign'] = np.log(df["CPI_Foreign"])


# In[58]:


df['Log_Nominal'] = np.log(df["Nominal_Exchange_Rate"])


# In[59]:


df['Log_Real'] = np.log(df["Real_Exchange_Rate"])


# In[60]:


print(df)


# In[61]:


print(ts)


# In[63]:


plt.bar(range(len(acf_result)), acf_result)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()


# In[64]:


pacf_result = sm.tsa.pacf(ts, nlags=20)

print(pacf_result)


# In[65]:


plt.bar(range(len(pacf_result)), pacf_result)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()


# In[66]:


import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# In[67]:


def adf_test(series, title=''):
    result = adfuller(series, autolag='AIC')
    output = {'Test Statistic': result[0], 'p-value': result[1], 
              'Lags Used': result[2], 'Number of Observations Used': result[3]}
    for key, value in result[4].items():
        output[f'Critical Value ({key})'] = value
    print(title)
    print('--------------------------------------')
    for key, value in output.items():
        print(f'{key}: {value:.4f}')
    print('\n')


# In[68]:


plt.figure(figsize=(12, 8))

variables = ['Log_Nominal', 'Log_Real', 'Log_CPI_Home', 'Log_CPI_Foreign']

for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    df[var].plot(title=var)
    plt.tight_layout()
    adf_test(df[var], title=var)

plt.show()


# In[78]:


df['Log_Real']


# In[117]:


df['diff_Log_Real'] = df['Log_Real'].diff().dropna()
df_new = df.dropna()


# In[118]:


print(df_new)


# In[119]:


plt.figure(figsize=(12, 8))

variables = ['diff_Log_Real']

for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    df_new[var].plot(title=var)
    plt.tight_layout()
    adf_test(df_new[var], title=var)

plt.show()


# In[120]:


acf_result = sm.tsa.acf(df_new['diff_Log_Real'], nlags=20)

print(acf_result)


# In[121]:


plt.bar(range(len(acf_result)), acf_result)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()


# In[113]:


pacf_result = sm.tsa.pacf(df_new['diff_Log_Real'], nlags=20)

print(pacf_result)


# In[114]:


plt.bar(range(len(pacf_result)), pacf_result)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# In[103]:


plt.figure(figsize=(12, 8))

variables = ['Real_Exchange_Rate']

for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    df[var].plot(title=var)
    plt.tight_layout()
    adf_test(df[var], title=var)

plt.show()


# In[105]:


df['diff_Real_Exchange_Rate'] = df['Real_Exchange_Rate'].diff().dropna()
df_new1 = df.fillna(df.mean())
print(df_new1)


# In[108]:


plt.figure(figsize=(12, 8))

variables = ['diff_Real_Exchange_Rate']

for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    df_new1[var].plot(title=var)
    plt.tight_layout()
    adf_test(df_new1[var], title=var)

plt.show()


# In[109]:


acf_result_Real = sm.tsa.acf(df_new1['diff_Real_Exchange_Rate'], nlags=20)

print(acf_result_Real)


# In[110]:


plt.bar(range(len(acf_result_Real)), acf_result_Real)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()


# In[115]:


pacf_result_Real = sm.tsa.pacf(df_new1['diff_Real_Exchange_Rate'], nlags=20)

print(pacf_result_Real)


# In[116]:


plt.bar(range(len(pacf_result_Real)), pacf_result_Real)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# In[130]:


df1 = df_new[['Date','diff_Log_Real']]


# In[133]:


train_size = int(0.8 * len(df1))
train_data = df1[:train_size]
test_data = df1[train_size:]


# In[134]:


model = ARIMA(train_data['diff_Log_Real'], order=(1, 0, 1))
results = model.fit()


# In[135]:


print(results.summary())


# In[136]:


results.plot_diagnostics()
plt.show()


# In[149]:


forecast = results.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast)


# In[152]:


actual = df1[103:131]
print(actual['diff_Log_Real'])


# In[154]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(actual['diff_Log_Real'], forecast)
mae


# In[164]:


model2 = ARIMA(train_data['diff_Log_Real'], order=(5, 0, 4))
results2 = model.fit()


# In[165]:


print(results2.summary())


# In[166]:


results2.plot_diagnostics()
plt.show()


# In[167]:


forecast2 = results2.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast2)


# In[ ]:





# In[ ]:





# In[ ]:




