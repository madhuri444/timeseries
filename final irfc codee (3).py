# important packages
from nsepy import get_history
from datetime import date
import pandas as pd
import numpy as np
import pickle
#import dtale as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns
import statsmodels.api as sm
import pylab as py
from statsmodels.tsa.stattools import adfuller
from numpy import log
from dateutil.relativedelta import *

df = get_history(symbol="IRFC",series = "N1",start=date(2015,1,1),end=date.today())
df = df.reset_index()
df.drop(['Symbol','Series','Turnover',"%Deliverble"],axis =1,inplace =True)

## Before EDA
## Dtale
#d1=dt.show(df)
#d1.open_browser()

### Code EDA
sns.pairplot(df)    ### correlation plot
### Maxmimum correlation is for LAST AND CLOSE column

### histogram
plt.hist(df["Prev Close"]);plt.xlabel('Prev Close')
plt.hist(df["Open"]);plt.xlabel('Open')
plt.hist(df['High']);plt.xlabel('High')
plt.hist(df['Low']);plt.xlabel('Low')
plt.hist(df['Last']);plt.xlabel('Last')
plt.hist(df['Close']);plt.xlabel('Close')
plt.hist(df['VWAP']);plt.xlabel('VWAP')
plt.hist(df['Volume']);plt.xlabel('Volume')
plt.hist(df['Trades']);plt.xlabel('Trades')
plt.hist(df['Deliverable Volume']);plt.xlabel('Deliverable Volume')

### QQ Plot
sm.qqplot(df["Prev Close"]);plt.xlabel('Prev Close')
sm.qqplot(df["Open"]);plt.xlabel('Open')
sm.qqplot(df['High']);plt.xlabel('High')
sm.qqplot(df['Low']);plt.xlabel('Low')
sm.qqplot(df['Last']);plt.xlabel('Last')
sm.qqplot(df['Close']);plt.xlabel('Close')
sm.qqplot(df['VWAP']);plt.xlabel('VWAP')
sm.qqplot(df['Volume']);plt.xlabel('Volume')
sm.qqplot(df['Trades']);plt.xlabel('Trades')
sm.qqplot(df['Deliverable Volume']);plt.xlabel('Deliverable Volume')

df['Date']= pd.to_datetime(df['Date'])


# Creating dummy dataframe for impulation
r = pd.date_range(start = df.Date.min(), end = df.Date.max())

# replace range with Date column in Data set and fill NA in remaining columns
dummy = df.set_index('Date').reindex(r).fillna(' ').rename_axis('Date').reset_index()


### Price imputation
dummy = dummy.replace(' ',np.nan)
dummy = dummy.ffill()


dummy.set_index('Date', inplace=True)

df = dummy

df = pd.DataFrame(df['Close'])
df.columns = ['Price']
df1=df
#### Dtale
#d=dt.show(df1)
#d.open_browser()

### After EDA
### Histogram
plt.hist(df);plt.xlabel('Open')
## QQ plot
sm.qqplot(df);plt.xlabel('Price')
### Seasonality plot
#seasonality -  additive

result = seasonal_decompose(df, model='addtive', freq = 30)
fig = plt.figure()
fig = result.plot()  
fig.set_size_inches(16, 9)


#seasonality -  multiplicative
result = seasonal_decompose(df, model='multiplicative', freq = 30)
fig = plt.figure()    
fig = result.plot()  
fig.set_size_inches(16, 9)




## model building
df.info()

### Parameters Selection
result = adfuller(df.Price.dropna())
result1 = adfuller(df["Price"].diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('p-value for 1st difference: %f' % result1[1])

#*PACF plot to find p value for ARIMA model*

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = df['Price'].diff()
plt.figure()
plot_pacf(data.dropna(), lags=40)
plt.xlabel('Lags', fontsize=12)
plt.ylabel('Partial Autocorrelation', fontsize=12)
plt.title('Partial Autocorrelation of First Order Differenced Series', fontsize=14)
plt.show()

# p = 1
"""Forecasting

*Augmented Dickey Fuller test*

If P Value > 0.05 we go ahead with finding the order of differencing.

If P Value < 0.05 then the data is indeed stationary no need to find order of differencing i.e; d=0.
"""

"""We take q(residuals) value as 1 beacause only residual lag 1 is able to cross significanve value(blue area)"""
#We take p(lags) value as 1 beacause only lag 1 is able to cross significanve value(blue area)


#*ACF plot to find q value for ARIMA model*

data = df['Price'].diff()
plt.figure()
plot_acf(data.dropna(), lags=40)
plt.xlabel('Reciduals', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.title('Autocorrelation of First Order Differenced Series', fontsize=14)
plt.show()

#q = 1

df_train = df.loc[:'2020-12-31', :]#loc last is inclusive
df_test = df.loc['2021-01-01':, :]

### Training model
arima_model = ARIMA(df_train, order=(1, 0, 0))
fitted = arima_model.fit(disp = -1)
print(fitted.summary())

# Forecast

fc, se, conf = fitted.forecast(len(df_test.Price), alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index = df_test.index)
lower_series = pd.Series(conf[:, 0], index=df_test.index)
upper_series = pd.Series(conf[:, 1], index=df_test.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(df_train, label='training')
plt.plot(df_test, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.05)
plt.title('SBIN Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()
ARIMA_acc = mean_squared_error(df_test.Price, fc_series)
print(ARIMA_acc)


#Main Forecast
arima_model = ARIMA(df, order=(1, 0, 0))
fitted = arima_model.fit(disp = -1)
print(fitted.summary())

#fc1, se1, conf1 = fitted.forecast(1826, alpha=0.05)  # 95% confidence
#fc_series1 = pd.DataFrame(fc1, index = pd.date_range(start = pd.to_datetime('today').date(), periods = 1826, freq='D'), columns = ['forecast'])
#lower_series1 = pd.Series(conf1[:, 0], index=pd.date_range(start = pd.to_datetime('today').date(), periods = 1826, freq='d'))
#upper_series1 = pd.Series(conf1[:, 1], index=pd.date_range(start = pd.to_datetime('today').date(), periods = 1826, freq='d'))

plt.figure(figsize=(12,5), dpi=100)
plt.plot(df, color = 'blue', label='Actual Price')
plt.plot(fc_series1, color = 'orange',label='Predicted Price')
plt.fill_between(lower_series1.index, lower_series1, upper_series1, 
                 color='k', alpha=.05)
plt.title('IRFC Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

pickle.dump(arima_model,open('model.pkl','wb'))
import os
os.getcwd()
