#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import statsmodels.api as sm


# In[9]:


df = pd.read_csv(r"C:\Users\eoinh\OneDrive\Documents\Master_Excel.csv")


# In[10]:


print(df)


# In[31]:


df['Log_CPI_Home'] = np.log(df["CPI_Home"])


# In[32]:


df['Log_CPI_Foreign'] = np.log(df["CPI_Foreign"])


# In[36]:


df['Log_Nominal'] = np.log(df["Nominal_Exchange_Rate"])


# In[37]:


df['Log_Real'] = np.log(df["Real_Exchange_Rate"])


# In[38]:


print(df)


# In[43]:


ts = pd.Series(df['Log_CPI_Home'].values, index=pd.date_range(start='2013-01-01', periods=len(df), freq='M'))

acf_result = sm.tsa.acf(ts, nlags=len(ts)-1)



# In[54]:


plt.bar(range(len(acf_result)), acf_result)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()


# In[55]:


plt.bar()


# In[ ]:




