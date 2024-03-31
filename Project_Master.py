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
Master_df = pd.read_csv("Master_Excel")
Master_df.head()

#PART_C
# Calculating the logarithms
Master_df['Log_Nominal_Exchange_Rate'] = np.log(Master_df['Nominal_Exchange_Rate'])
Master_df['Log_Real_Exchange_Rate'] = np.log(Master_df['Real_Exchange_Rate'])
Master_df['Log_CPI_Home'] = np.log(Master_df['CPI_Home'])
Master_df['Log_CPI_Foreign'] = np.log(Master_df['CPI_Foreign'])

Master_df.head()  # Display the first few rows of the dataset to check the calculation

