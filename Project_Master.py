import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pip install yfinance

import yfinance as yf
import pandas as pd

# Define the currency pair (USDINR=X for USD to INR)
currency_pair = 'CHFINR=X'

# Define the time period for which you want the data
start_date = '2012-12-1'
end_date = '2023-12-1'

# Fetching the data
data = yf.download(currency_pair, start=start_date, end=end_date)

# Resampling the data to get the last data point of each month
monthly_data = data.resample('M').last()

# Displaying the first few rows of the data
monthly_data

# Reset the index to turn it into a normal range index
monthly_data.reset_index(inplace=True)

# Save the data to an Excel file(Dont run this code)
excel_file_name = 'Swiss_INR_Exchange_Rate_Monthly.xlsx'
monthly_data.to_excel(excel_file_name, index=False)  # Set index=False to not save the new range index

# Confirming that the file has been saved(Dont run this code)
print(f'Data saved to {excel_file_name}')

#Import master file
Master_df = pd.read_excel('Master Excel.xls')
Master_df.head()

#PART_C
# Calculating the logarithms
Master_df['Log_Nominal_Exchange_Rate'] = np.log(Master_df['Nominal_Exchange_Rate'])
Master_df['Log_Real_Exchange_Rate'] = np.log(Master_df['Real_Exchange_Rate'])
Master_df['Log_CPI_Home'] = np.log(Master_df['CPI_Home'])
Master_df['Log_CPI_Foreign'] = np.log(Master_df['CPI_Foreign'])

Master_df.head()  # Display the first few rows of the dataset to check the calculation

# PART_D
# Analyze the properties of these variables (e.g., trends, seasonality, stationarity).
# For stationarity testing, we can use the Augmented Dickey-Fuller (ADF) test.
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Function to perform Augmented Dickey-Fuller test
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

# Plotting the series and performing ADF test
plt.figure(figsize=(12, 8))

variables = ['Log_Nominal_Exchange_Rate', 'Log_Real_Exchange_Rate', 'Log_CPI_Home', 'Log_CPI_Foreign']

for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    Master_df[var].plot(title=var)
    plt.tight_layout()
    adf_test(Master_df[var], title=var)

plt.show()


df['diff_log_CPI_foreign'] = df['log_CPI_foreign'].diff().dropna()

