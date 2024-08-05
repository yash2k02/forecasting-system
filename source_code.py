import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
sns.set_style('darkgrid')
import matplotlib


## Reading the data
df = pd.read_excel('hist_data.xlsx')
df
df.shape
df.info()
df = df.set_index(['Year'])
df.index.min(), df.index.max()



mean_monthlysales = df['Monthly_Sales'].resample('MS').mean()
print(mean_monthlysales)
mean_monthlysales.plot(figsize=(15,5),c='green')
plt.show()


## Data Pre-Processing (top 3 and bottom 2 products)
df['Product'].unique()
df.isna().sum()
df


### Toiletpaper
df = pd.read_excel('hist_data.xlsx')
toiletpaper = df.loc[df['Product'] == 'TOILET PAPER']
print(toiletpaper['Year'].min(), toiletpaper['Year'].max())
toiletpaper = toiletpaper.sort_values('Year')
toiletpaper
toiletpaper.isnull().sum()
toiletpaper = toiletpaper.groupby('Year')['Monthly_Sales'].sum().reset_index()
toiletpaper
toiletpaper = toiletpaper.set_index('Year')
toiletpaper
toiletpaper.index

toiletpaper.plot(figsize=(20,5))
plt.title("Toilet Paper sales",fontsize=16, fontweight='bold')
toiletpaper
toiletpaper.index = pd.to_datetime(toiletpaper.index)
y_tp = toiletpaper['Monthly_Sales'].resample('MS').mean()
y_tp
y_tp.plot(figsize=(14,6))
plt.title('Mean Monthly Sales of Toilet Paper', fontsize=16, fontweight='bold')
plt.ylabel("Sales")
plt.show()
fig = plt.figure(figsize=(8,6))
sns.boxplot(y_tp).set_title('Box Plot on sales of Toilet Paper')
plt.show()



### Chitos
chitos = df.loc[df['Product'] == 'CHITOS']
chitos
print(chitos['Year'].min(), chitos['Year'].max())

chitos = chitos.sort_values('Year')
chitos
chitos.isnull().sum()

chitos = chitos.groupby('Year')['Monthly_Sales'].sum().reset_index()
chitos
chitos = chitos.set_index('Year')
chitos

chitos.index

chitos.plot(figsize=(20,5))
plt.title("Chitos Sales", fontsize=16, fontweight='bold')
chitos.index = pd.to_datetime(chitos.index)
y_ch = chitos['Monthly_Sales'].resample('MS').mean()
y_ch
y_ch.plot(figsize=(14,6))
plt.title('Mean Monthly Sales of Chitos', fontsize=16, fontweight='bold')
plt.ylabel("Sales")
plt.show()
fig = plt.figure(figsize=(8,6))
sns.boxplot(y_ch).set_title('Box Plot on sales of Chitos')
plt.show()




### Peanut Butter
peanutbutter = df.loc[df['Product'] == ' PEANUT BUTTER']
peanutbutter
print(peanutbutter['Year'].min(), peanutbutter['Year'].max())
peanutbutter = peanutbutter.sort_values('Year')
peanutbutter
peanutbutter.isnull().sum()

peanutbutter = peanutbutter.groupby('Year')['Monthly_Sales'].sum().reset_index()
peanutbutter

peanutbutter = peanutbutter.set_index('Year')
peanutbutter

peanutbutter.index

peanutbutter.plot(figsize=(20,5))
plt.title("Peanut Butter Sales", fontsize=16, fontweight='bold')
peanutbutter.index = pd.to_datetime(peanutbutter.index)
y_pb = peanutbutter['Monthly_Sales'].resample('MS').mean()
y_pb
y_pb.plot(figsize=(14,6))
plt.title('Mean Monthly Sales of Peanut Butter', fontsize=16, fontweight='bold')
plt.ylabel("Sales")
plt.show()
fig = plt.figure(figsize=(8,6))
sns.boxplot(y_pb).set_title('Box Plot on sales of Peanut Butter')
plt.show()



### Conditioner
conditioner = df.loc[df['Product'] == 'CONDITIONER']
conditioner
print(conditioner['Year'].min(), conditioner['Year'].max())
conditioner = conditioner.sort_values('Year')
conditioner
conditioner.isnull().sum()

conditioner = conditioner.groupby('Year')['Monthly_Sales'].sum().reset_index()
conditioner
conditioner = conditioner.set_index('Year')
conditioner
conditioner.index

conditioner.plot(figsize=(20,5))
plt.title("Conditioner Sales", fontsize=16, fontweight='bold')
conditioner.index = pd.to_datetime(conditioner.index)
y_co = conditioner['Monthly_Sales'].resample('MS').mean()
y_co
y_co.plot(figsize=(14,6))
plt.title('Mean Monthly Sales of Conditioner', fontsize=16, fontweight='bold')
plt.ylabel("Sales")
plt.show()
fig = plt.figure(figsize=(8,6))
sns.boxplot(y_co).set_title('Box Plot on sales of Conditioner')
plt.show()



### Flour
flour = df.loc[df['Product'] == 'FLOUR']
flour
print(flour['Year'].min(), flour['Year'].max())
flour = flour.sort_values('Year')
flour
flour.isnull().sum()

flour = flour.groupby('Year')['Monthly_Sales'].sum().reset_index()
flour
flour = flour.set_index('Year')
flour
flour.index

flour.plot(figsize=(20,5))
plt.title("Flour Sales", fontsize=16, fontweight='bold')
flour.index = pd.to_datetime(flour.index)
y_fl = flour['Monthly_Sales'].resample('MS').mean()
y_fl
y_fl.plot(figsize=(14,6))
plt.title('Mean Monthly Sales of Flour', fontsize=16, fontweight='bold')
plt.ylabel("Sales")
plt.show()
fig = plt.figure(figsize=(8,6))
sns.boxplot(y_fl).set_title('Box Plot on sales of Flour')
plt.show()


## ETS decomposition
### Toilerpaper
rcParams['figure.figsize'] = 12,8
decomposition_tp = sm.tsa.seasonal_decompose(y_tp, model='additive')
fig = decomposition_tp.plot()
plt.show()
### Chitos
rcParams['figure.figsize'] = 12,8
decomposition_ch = sm.tsa.seasonal_decompose(y_ch, model='additive')
fig = decomposition_ch.plot()
plt.show()
### Peanut Butter
rcParams['figure.figsize'] = 12,8
decomposition_pb = sm.tsa.seasonal_decompose(y_pb, model='additive')
fig = decomposition_pb.plot()
plt.show()
### Conditioner
rcParams['figure.figsize'] = 12,8
decomposition_co = sm.tsa.seasonal_decompose(y_co, model='additive')
fig = decomposition_co.plot()
plt.show()
### Flour
rcParams['figure.figsize'] = 12,8
decomposition_fl = sm.tsa.seasonal_decompose(y_fl, model='additive')
fig = decomposition_fl.plot()
plt.show()



## Test for stationarity

def adfuller_test(series):
    result = adfuller(series)
    labels = ['ADF test statistic', 'p-value', 'Number of observations used', '#lags used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
adfuller_test(y_tp)
adfuller_test(y_ch)
adfuller_test(y_pb)
adfuller_test(y_co)
adfuller_test(y_fl)


#### Here the approach used to find optimal parameters for SARIMA model is by using sarimax function of statsmodel.

### Toilet Paper
# Define the p, d and q parameters to take any value between 0 and 3
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_tp = sm.tsa.statespace.SARIMAX(y_tp,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results_tp = mod_tp.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_tp.aic))
        except:
          continue



mod_tp = sm.tsa.statespace.SARIMAX(y_tp,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_invertibility=False)
results_tp = mod_tp.fit()
print(results_tp.summary().tables[1])

#### Model diagnosis
results_tp.plot_diagnostics(figsize=(12, 8))
plt.show()

#### Visualising Forecasts
pred = results_tp.get_prediction(start=pd.to_datetime('2021-01-01'), dynamic = False)
pred_ci = pred.conf_int()
ax = y_tp['2018':].plot(label = 'observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast' , alpha = .7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha= .2)
plt.title('Forecast of Toilet Paper Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Toilet Paper Sales')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y_tp['2021-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results_tp.get_forecast(steps = 36)
pred_ci = pred_uc.conf_int()
ax = y_tp.plot(label='observed', figsize = (12, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = .25)
plt.title('Forecast of Toilet Paper Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Toilet Paper Sales')
plt.legend()
plt.show()


### Chitos
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_ch = sm.tsa.statespace.SARIMAX(y_ch,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results_ch = mod_ch.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_ch.aic))
        except:
          continue


mod_ch = sm.tsa.statespace.SARIMAX(y_ch,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_invertibility=False)
results_ch = mod_ch.fit()
print(results_ch.summary().tables[1])
results_ch.plot_diagnostics(figsize=(12, 8))
plt.show()
pred = results_ch.get_prediction(start=pd.to_datetime('2021-01-01'), dynamic = False)
pred_ci = pred.conf_int()
ax = y_ch['2018':].plot(label = 'observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast' , alpha = .7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha= .2)
plt.title('Forecast of Chitos Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Chitos Sales')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y_ch['2021-01-01' :]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results_ch.get_forecast(steps = 36)
pred_ci = pred_uc.conf_int()
ax = y_ch.plot(label='observed', figsize = (12, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = .25)
plt.title('Forecast of Chitos Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Chitos Sales')
plt.legend()
plt.show()


### Peanut Butter
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_pb = sm.tsa.statespace.SARIMAX(y_pb,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results_pb = mod_pb.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_pb.aic))
        except:
          continue

mod_pb = sm.tsa.statespace.SARIMAX(y_pb,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_invertibility=False)
results_pb = mod_pb.fit()
print(results_pb.summary().tables[1])
results_pb.plot_diagnostics(figsize=(12, 8))
plt.show()
pred = results_pb.get_prediction(start=pd.to_datetime('2021-01-01'), dynamic = False)
pred_ci = pred.conf_int()
ax = y_pb['2018':].plot(label = 'observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast' , alpha = .7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha= .2)
plt.title('Forecast of Peanut Butter Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Peanut Butter Sales')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y_pb['2021-01-01' :]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results_pb.get_forecast(steps = 36)
pred_ci = pred_uc.conf_int()
ax = y_pb.plot(label='observed', figsize = (12, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = .25)
plt.title('Forecast of Peanut Butter Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Peanut Butter Sales')
plt.legend()
plt.show()


### Conditioner
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_co = sm.tsa.statespace.SARIMAX(y_co,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results_co = mod_co.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_co.aic))
        except:
          continue

mod_co = sm.tsa.statespace.SARIMAX(y_co,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_invertibility=False)
results_co = mod_co.fit()
print(results_co.summary().tables[1])
results_co.plot_diagnostics(figsize=(12, 8))
plt.show()
pred = results_co.get_prediction(start=pd.to_datetime('2021-01-01'), dynamic = False)
pred_ci = pred.conf_int()
ax = y_co['2018':].plot(label = 'observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast' , alpha = .7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha= .2)
plt.title('Forecast of Conditioner Sales', fontsize=16, fontweight='bold')

ax.set_xlabel('Year')
ax.set_ylabel('Conditioner Sales')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y_co['2021-01-01' :]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results_co.get_forecast(steps = 36)
pred_ci = pred_uc.conf_int()
ax = y_co.plot(label='observed', figsize = (12, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = .25)
plt.title('Forecast of Conditioner Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Conditioner Sales')
plt.legend()
plt.show()


### Flour
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_fl = sm.tsa.statespace.SARIMAX(y_fl,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results_fl = mod_fl.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_fl.aic))
        except:
          continue

mod_fl = sm.tsa.statespace.SARIMAX(y_fl,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_invertibility=False)
results_fl = mod_fl.fit()
print(results_fl.summary().tables[1])
results_fl.plot_diagnostics(figsize=(12, 8))
plt.show()
pred = results_fl.get_prediction(start=pd.to_datetime('2021-01-01'), dynamic = False)
pred_ci = pred.conf_int()
ax = y_fl['2018':].plot(label = 'observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast' , alpha = .7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha= .2)
plt.title('Forecast of Flour Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Flour Sales')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y_fl['2021-01-01' :]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results_fl.get_forecast(steps = 36)
pred_ci = pred_uc.conf_int()
ax = y_fl.plot(label='observed', figsize = (12, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = .25)
plt.title('Forecast of Flour Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Flour Sales')
plt.legend()
plt.show()



## Comparing Products over historical sales
### Toiletpaper vs Chitos Sales
toiletpaper = pd.DataFrame({'Year':y_tp.index, 'Monthly_Sales':y_tp.values})
toiletpaper.head()
chitos = pd.DataFrame({'Year':y_ch.index, 'Monthly_Sales': y_ch.values})
chitos.head()
store = toiletpaper.merge(chitos,how= 'inner',on='Year')
store.rename(columns={'Monthly_Sales_x': 'toiletpaper_sales', 'Monthly_Sales_y': 'chitos_sales'}, inplace = True)
store.head()

plt.figure(figsize=(20,8))
plt.plot(store['Year'], store['toiletpaper_sales'], 'b-', label = 'toiletpaper')
plt.plot(store['Year'], store['chitos_sales'], 'g-', label = 'chitos')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Toilet Paper vs Chitos Sales')
plt.legend(["toiletpaper","chitos"],loc= "upper right")


### Peanut Butter vs Conditioner Sales

peanut_butter = pd.DataFrame({'Year':y_pb.index, 'Monthly_Sales':y_pb.values})
peanut_butter.head()
conditioner = pd.DataFrame({'Year':y_co.index, 'Monthly_Sales':y_co.values})
conditioner.head()
store = peanut_butter.merge(conditioner,how= 'inner',on='Year')
store.rename(columns={'Monthly_Sales_x': 'peanut_butter_sales', 'Monthly_Sales_y': 'conditioner_sales'}, inplace = True)
store.head()

plt.figure(figsize=(20,8))
plt.plot(store['Year'], store['peanut_butter_sales'], 'b-', label = 'peanut_butter')
plt.plot(store['Year'], store['conditioner_sales'], 'g-', label = 'conditioner')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Peanut Butter vs Conditioner Sales')
plt.legend(["peanut butter","conditioner"],loc= "upper right")

### Toilet Paper vs Flour Sales

toiletpaper = pd.DataFrame({'Year':y_tp.index, 'Monthly_Sales': y_tp.values})
toiletpaper.head()
flour = pd.DataFrame({'Year':y_fl.index, 'Monthly_Sales':y_fl.values})
flour.head()
store = toiletpaper.merge(flour,how= 'inner',on='Year')
store.rename(columns={'Monthly_Sales_x': 'toiletpaper_sales', 'Monthly_Sales_y': 'flour_sales'}, inplace = True)
store.head()

plt.figure(figsize=(20,8))
plt.plot(store['Year'], store['toiletpaper_sales'], 'g-', label = 'toiletpaper')
plt.plot(store['Year'], store['flour_sales'], 'b-', label = 'flour')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Toilet Paper vs Flour Sales')
plt.legend(["toiletpaper","flour"],loc= "upper right")


### Conditioner vs Flour Sales
conditioner = pd.DataFrame({'Year':y_co.index, 'Monthly_Sales':y_co.values})
conditioner.head()
flour = pd.DataFrame({'Year':y_fl.index, 'Monthly_Sales': y_fl.values})
flour.head()
store = conditioner.merge(flour,how= 'inner',on='Year')
store.rename(columns={'Monthly_Sales_x': 'conditioner_sales', 'Monthly_Sales_y': 'flour_sales'}, inplace = True)
store.head()

plt.figure(figsize=(20,8))
plt.plot(store['Year'], store['conditioner_sales'], 'b-', label = 'conditioner')
plt.plot(store['Year'], store['flour_sales'], 'g-', label = 'flour')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Conditioner vs Flour Sales')
plt.legend(["conditioner","flour"],loc= "upper right")


## Forecasting Products using Fbprophet library
### Toilet Paper

toiletpaper = toiletpaper.rename(columns={'Year':'ds', 'Monthly_Sales': 'y'})
toiletpaper
import prophet
toiletpaper_model = prophet.Prophet(interval_width=0.95) 
toiletpaper_model.fit(toiletpaper)

toiletpaper_forecast = toiletpaper_model.make_future_dataframe(periods=36, freq='MS')
toiletpaper_forecast = toiletpaper_model.predict(toiletpaper_forecast)
toiletpaper_forecast
toiletpaper_forecast.columns

plt.figure(figsize=(20, 10))
toiletpaper_model.plot(toiletpaper_forecast, xlabel='Year', ylabel='Sales')
plt.title('Toilet Paper Sales')
plt.show()


### Chitos

chitos = chitos.rename(columns={'Year': 'ds', 'Monthly_Sales': 'y'})
chitos
chitos_model = prophet.Prophet(interval_width=0.95)  
chitos_model.fit(chitos)
chitos_forecast = chitos_model.make_future_dataframe(periods=36, freq='MS')
chitos_forecast = chitos_model.predict(chitos_forecast)
chitos_forecast
chitos_forecast.columns
plt.figure(figsize=(20, 10))
chitos_model.plot(chitos_forecast, xlabel='Year', ylabel='Sales')
plt.title('Chitos Sales')
plt.show()


### Peanut Butter

peanut_butter = peanut_butter.rename(columns={'Year': 'ds', 'Monthly_Sales': 'y'})
peanut_butter
peanut_butter_model = prophet.Prophet(interval_width=0.95) 
peanut_butter_model.fit(peanut_butter)
peanut_butter_forecast = peanut_butter_model.make_future_dataframe(periods=36, freq='MS')
peanut_butter_forecast = peanut_butter_model.predict(peanut_butter_forecast)
peanut_butter_forecast
peanut_butter_forecast.columns
plt.figure(figsize=(20, 10))
peanut_butter_model.plot(peanut_butter_forecast, xlabel='Year', ylabel='Sales')
plt.title('Peanut Butter Sales')
plt.show()

plot_components_plotly(peanut_butter_model , peanut_butter_forecast)


### Conditioner

conditioner = conditioner.rename(columns={'Year': 'ds', 'Monthly_Sales': 'y'})
conditioner
conditioner_model = prophet.Prophet(interval_width=0.95)  # Create an instance of the Prophet class
conditioner_model.fit(conditioner)
conditioner_forecast = conditioner_model.make_future_dataframe(periods=36, freq='MS')
conditioner_forecast = conditioner_model.predict(conditioner_forecast)
conditioner_forecast
conditioner_forecast.columns
plt.figure(figsize=(20, 10))
chitos_model.plot(conditioner_forecast, xlabel='Year', ylabel='Sales')
plt.title('Conditioner Sales')
plt.show()


### Flour

flour = chitos.rename(columns={'Year': 'ds', 'Monthly_Sales': 'y'})
flour
flour_model = prophet.Prophet(interval_width=0.95) 
flour_model.fit(flour)
flour_forecast = flour_model.make_future_dataframe(periods=36, freq='MS')
flour_forecast = flour_model.predict(flour_forecast)
flour_forecast
flour_forecast.columns
plt.figure(figsize=(20, 10))
chitos_model.plot(flour_forecast, xlabel='Year', ylabel='Sales')
plt.title('Flour Sales')
plt.show()


## Comparing Forecast and Trend Visualization
### Toilet Paper vs Chitos

chitos_names = ['chitos_%s' % column for column in chitos_forecast.columns]
toiletpaper_names = ['toiletpaper%s' % column for column in toiletpaper_forecast.columns]


merge_chitos_forecast = chitos_forecast.copy()
merge_toiletpaper_forecast = toiletpaper_forecast.copy()


merge_chitos_forecast.columns = chitos_names
merge_toiletpaper_forecast.columns = toiletpaper_names


forecast1 = pd.merge(merge_chitos_forecast, merge_toiletpaper_forecast, how = 'inner', left_on = 'chitos_ds', right_on = 'toiletpaperds')

forecast1 = forecast1.rename(columns={'chitos_ds': 'Date'}).drop('toiletpaperds', axis=1)
forecast1.head()

plt.figure(figsize=(15, 7))
plt.plot(forecast1['Date'], forecast1['toiletpaperyhat'], 'b-')
plt.plot(forecast1['Date'], forecast1['chitos_yhat'], 'g-')
plt.legend(['toiletpaper','chitos'])
plt.xlabel('Year')
plt.ylabel('Sales')
plt.tick_params(axis='x', labelsize=16)
plt.title('Toilet Paper v/s Chitos Sales Forecast')
plt.figure(figsize=(11, 7))
plt.plot(forecast1['Date'], forecast1['toiletpapertrend'], 'b-')
plt.plot(forecast1['Date'], forecast1['chitos_trend'], 'g-')

plt.legend(['toiletpaper','chitos'])
plt.xlabel('Year'); plt.ylabel('Sales')
plt.title('Toilet paper and Chitos Sales Trend')

### Peanut Butter vs Chitos

peanut_butter_names = ['peanut_butter%s' % column for column in peanut_butter_forecast.columns]
chitos_names = ['chitos%s' % column for column in chitos_forecast.columns]

merge_peanut_butter_forecast = peanut_butter_forecast.copy()
merge_chitos_forecast = chitos_forecast.copy()

merge_peanut_butter_forecast.columns = peanut_butter_names
merge_chitos_forecast.columns = chitos_names

forecast2 = pd.merge(merge_peanut_butter_forecast, merge_chitos_forecast, how = 'inner', left_on = 'peanut_butterds', right_on = 'chitosds')

forecast2 = forecast2.rename(columns={'peanut_butterds': 'Date'}).drop('chitosds', axis=1)
forecast2.head()

plt.figure(figsize=(15, 7))
plt.plot(forecast2['Date'], forecast2['peanut_butteryhat'], 'b-')
plt.plot(forecast2['Date'], forecast2['chitosyhat'], 'g-')
plt.legend(['peanutbutter','chitos'])
plt.xlabel('Year')
plt.ylabel('Sales')
plt.tick_params(axis='x', labelsize=16)
plt.title('Peanut Butter  v/s Chitos Sales Forecast');
plt.figure(figsize=(11, 7))
plt.plot(forecast2['Date'], forecast2['peanut_buttertrend'], 'b-')
plt.plot(forecast2['Date'], forecast2['chitostrend'], 'g-')

plt.legend(['peanut_butter','chitos Supplies'])
plt.xlabel('Year'); plt.ylabel('Sales')
plt.title('Peanut Butter and Chitos Sales Trend')

### Conditioner vs Flour

flour_names = ['flour_%s' % column for column in flour_forecast.columns]
conditioner_names = ['conditioner%s' % column for column in conditioner_forecast.columns]


merge_flour_forecast = flour_forecast.copy()
merge_conditioner_forecast = conditioner_forecast.copy()


merge_flour_forecast.columns = flour_names
merge_conditioner_forecast.columns = conditioner_names


forecast3 = pd.merge(merge_flour_forecast, merge_conditioner_forecast, how = 'inner', left_on = 'flour_ds', right_on = 'conditionerds')

forecast3 = forecast3.rename(columns={'flour_ds': 'Date'}).drop('conditionerds', axis=1)
forecast3.head()
plt.figure(figsize=(15, 7))
plt.plot(forecast3['Date'], forecast3['conditioneryhat'], 'b-')
plt.plot(forecast3['Date'], forecast3['flour_yhat'], 'g-')
plt.legend(['conditioner','flour'])
plt.xlabel('Year')
plt.ylabel('Sales')
plt.tick_params(axis='x', labelsize=16)
plt.title('Conditioner v/s Flour Sales Forecast');
plt.figure(figsize=(11, 7))
plt.plot(forecast3['Date'], forecast3['conditionertrend'], 'b-')
plt.plot(forecast3['Date'], forecast3['flour_trend'], 'g-')

plt.legend(['conditioner','flour'])
plt.xlabel('Year'); plt.ylabel('Sales')
plt.title('Conditioner and Flour Sales Trend')


### Toilet Paper vs Conditioner
conditioner_names = ['conditioner%s' % column for column in conditioner_forecast.columns]
toiletpaper_names = ['toiletpaper%s' % column for column in toiletpaper_forecast.columns]


merge_conditioner_forecast = conditioner_forecast.copy()
merge_toiletpaper_forecast = toiletpaper_forecast.copy()


merge_conditioner_forecast.columns = conditioner_names
merge_toiletpaper_forecast.columns = toiletpaper_names


forecast4 = pd.merge(merge_conditioner_forecast, merge_toiletpaper_forecast, how = 'inner', left_on = 'conditionerds', right_on = 'toiletpaperds')

forecast4 = forecast4.rename(columns={'conditionerds': 'Date'}).drop('toiletpaperds', axis=1)
forecast4.head()

plt.figure(figsize=(15, 7))
plt.plot(forecast4['Date'], forecast4['toiletpaperyhat'], 'b-')
plt.plot(forecast4['Date'], forecast4['conditioneryhat'], 'g-')
plt.legend(['toiletpaper','conditioner'])
plt.xlabel('Year'); plt.ylabel('Sales')
plt.tick_params(axis='x', labelsize=16)
plt.title('Toilet Paper v/s Conditioner Sales Forecast')

plt.figure(figsize=(11, 7))
plt.plot(forecast4['Date'], forecast4['toiletpapertrend'], 'b-')
plt.plot(forecast4['Date'], forecast4['conditionertrend'], 'g-')

plt.legend(['Toilet Paper','Conditioner'])
plt.xlabel('Year'); plt.ylabel('Sales')
plt.title('Toilet Paper and Conditioner Trend')
