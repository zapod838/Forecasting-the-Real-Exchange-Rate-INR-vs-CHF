#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt


# In[9]:


df = pd.read_csv(r"C:\Users\eoinh\OneDrive\Documents\Master_Excel.csv")


# In[10]:


print(df)


# In[19]:


Log_CPI_Home = np.log(df["CPI_Home"])
print(Log_CPI_Home)


# In[20]:


Log_CPI_Foreign = np.log(df["CPI_Foreign"])
print(Log_CPI_Foreign)


# In[22]:


Log_Nominal = np.log(df["Nominal_Exchange_Rate"])
print(Log_Nominal)


# In[23]:


Log_Real = np.log(df["Real_Exchange_Rate"])
print(Log_Real)


# In[ ]:




