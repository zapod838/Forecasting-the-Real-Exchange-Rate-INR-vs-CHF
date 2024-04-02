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


# In[194]:


model2 = ARIMA(train_data['diff_Log_Real'], order=(5, 0, 8))
results2 = model2.fit()


# In[195]:


print(results2.summary())


# In[196]:


results2.plot_diagnostics()
plt.show()


# In[197]:


forecast2 = results2.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast2)


# In[198]:


mae1 = mean_absolute_error(actual['diff_Log_Real'], forecast2)
mae1


# In[201]:


df2 = df_new[['Date','Log_Real']]
df2


# In[202]:


train_size2 = int(0.8 * len(df2))
train_data2 = df2[:train_size2]
test_data2 = df2[train_size2:]


# In[215]:


model3 = ARIMA(train_data2['Log_Real'], order=(1, 1, 1))
results3 = model3.fit()


# In[217]:


print(results3.summary())


# In[218]:


results3.plot_diagnostics()
plt.show()


# In[219]:


forecast3 = results3.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast3)


# In[220]:


actual3 = df[103:131]
print(actual3['Log_Real'])


# In[223]:


mae3 = mean_absolute_error(actual3['Log_Real'], forecast3)
mae3


# In[224]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual3['Log_Real'], forecast3)

# Calculate MSE
mse = mean_squared_error(actual3['Log_Real'], forecast3)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual3['Log_Real'] - forecast3) / actual3['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[225]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual['diff_Log_Real'], forecast)

# Calculate MSE
mse = mean_squared_error(actual['diff_Log_Real'], forecast)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual['diff_Log_Real'] - forecast) / actual['diff_Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[226]:


model4 = ARIMA(train_data2['Log_Real'], order=(3, 1, 4))
results4 = model4.fit()


# In[227]:


print(results4.summary())


# In[228]:


results4.plot_diagnostics()
plt.show()


# In[229]:


forecast4 = results4.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast4)


# In[230]:


actual4 = df[103:131]
print(actual4['Log_Real'])


# In[231]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual4['Log_Real'], forecast4)

# Calculate MSE
mse = mean_squared_error(actual4['Log_Real'], forecast4)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual4['Log_Real'] - forecast4) / actual4['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[232]:


model5 = ARIMA(train_data2['Log_Real'], order=(6, 1, 7))
results5 = model5.fit()


# In[233]:


print(results5.summary())


# In[234]:


results5.plot_diagnostics()
plt.show()


# In[235]:


forecast5 = results5.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast5)


# In[236]:


actual5 = df[103:131]
print(actual5['Log_Real'])


# In[237]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual5['Log_Real'], forecast5)

# Calculate MSE
mse = mean_squared_error(actual5['Log_Real'], forecast5)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual5['Log_Real'] - forecast5) / actual5['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[238]:


model6 = ARIMA(train_data2['Log_Real'], order=(3, 1, 2))
results6 = model6.fit()


# In[ ]:


print(results6.summary())


# In[239]:


results6.plot_diagnostics()
plt.show()


# In[240]:


forecast6 = results6.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast6)


# In[241]:


actual6 = df[103:131]
print(actual6['Log_Real'])


# In[242]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual6['Log_Real'], forecast6)

# Calculate MSE
mse = mean_squared_error(actual6['Log_Real'], forecast6)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual6['Log_Real'] - forecast6) / actual6['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[243]:


model7 = ARIMA(train_data2['Log_Real'], order=(3, 1, 6))
results7 = model7.fit()


# In[244]:


print(results7.summary())


# In[245]:


results7.plot_diagnostics()
plt.show()


# In[246]:


forecast7 = results7.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast7)


# In[247]:


actual7 = df[103:131]
print(actual7['Log_Real'])


# In[250]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual7['Log_Real'], forecast7)

# Calculate MSE
mse = mean_squared_error(actual7['Log_Real'], forecast7)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual7['Log_Real'] - forecast7) / actual7['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[251]:


model8 = ARIMA(train_data2['Log_Real'], order=(2, 1, 4))
results8 = model8.fit()


# In[253]:


print(results8.summary())


# In[254]:


results8.plot_diagnostics()
plt.show()


# In[255]:


forecast8 = results8.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast8)


# In[256]:


actual8 = df[103:131]
print(actual8['Log_Real'])


# In[259]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual8['Log_Real'], forecast8)

# Calculate MSE
mse = mean_squared_error(actual8['Log_Real'], forecast8)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual8['Log_Real'] - forecast8) / actual8['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[260]:


model9 = ARIMA(train_data2['Log_Real'], order=(4, 1, 4))
results9 = model9.fit()


# In[261]:


print(results9.summary())


# In[262]:


results9.plot_diagnostics()
plt.show()


# In[263]:


forecast9 = results9.forecast(steps=28)  # Example forecast for next 10 periods
print("Forecasted values:", forecast9)


# In[264]:


actual9 = df[103:131]
print(actual9['Log_Real'])


# In[265]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual9['Log_Real'], forecast9)

# Calculate MSE
mse = mean_squared_error(actual9['Log_Real'], forecast9)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual9['Log_Real'] - forecast9) / actual9['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[266]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE
mae = mean_absolute_error(actual4['Log_Real'], forecast4)

# Calculate MSE
mse = mean_squared_error(actual4['Log_Real'], forecast4)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np.mean(np.abs((actual4['Log_Real'] - forecast4) / actual4['Log_Real'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[ ]:




