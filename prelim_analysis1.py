# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:54:38 2020

@author: ssingh17
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import seaborn as sns

'''State Wise - Corn Production Share for 2010-2018'''
corn_prod = pd.read_csv('M:\Documents\Thesis\Preliminary Analysis\Corn Production - state level.csv')
corn_prod = corn_prod[(corn_prod['Year'] >= 2010) & (corn_prod['Year'] <= 2018) & (corn_prod['Period'] == 'YEAR')]
corn_prod['State'] = corn_prod['State'].str.title()
corn_prod = corn_prod.rename(columns={'Value': 'Production (BU)'})
corn_prod.head()
corn_prod['Year'].unique()
corn_prod =corn_prod[['State', 'Production (BU)']]
corn_prod_avg = corn_prod.groupby(['State'], as_index = False).mean()
corn_prod_avg['corn_prod_share'] = (corn_prod_avg['Production (BU)'] / corn_prod_avg['Production (BU)'].sum()) * 100
corn_prod_avg = corn_prod_avg.sort_values(['corn_prod_share'], ascending = [0])
corn_prod_avg.head(12)

'Corn production share pie chart'
labels = 'Iowa', 'Illinois', 'Nebraska', 'Minnesota', 'Indiana', 'South Dakota', 'Others'
sizes = [17.5, 15.1, 11.8, 10, 6.7, 5.3, 33.6]
explode = (0.1, 0, 0, 0, 0, 0, 0)  # only "explode" the 1st slice
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

'''State Wise - Corn Yield Share for 2010-2018'''
corn_yield = pd.read_csv('M:\Documents\Thesis\Preliminary Analysis\Corn Yield - state level.csv')
corn_yield = corn_yield[(corn_yield['Year'] >= 2010) & (corn_yield['Year'] <= 2018) & (corn_yield['Period'] == 'YEAR')]
corn_yield['State'] = corn_yield['State'].str.title()
corn_yield = corn_yield.rename(columns={'Value': 'Yield (BU per Acre)'})
corn_yield.head()
corn_yield['Year'].unique()
corn_yield =corn_yield[['State', 'Yield (BU per Acre)']]
corn_yield_avg = corn_yield.groupby(['State'], as_index = False).mean()
corn_yield_avg = corn_yield_avg.sort_values(['Yield (BU per Acre)'], ascending = [0])
corn_yield_avg.head(12)

'''Find the county which has highest avg yield in Iowa (the highest corn producing state in the US) for 2010-2018'''
corn_yield_IA = pd.read_csv('M:\Documents\Thesis\Preliminary Analysis\county_yield.csv')
corn_yield_IA = corn_yield_IA[(corn_yield_IA['Year'] >= 2010) & (corn_yield_IA['Year'] <= 2018) & (corn_yield_IA['State'] == 'IOWA')]
corn_yield_IA['County'] = corn_yield_IA['County'].str.title()
corn_yield_IA = corn_yield_IA.rename(columns={'Value': 'Yield (BU per Acre)'})
corn_yield_IA.head()
corn_yield_IA['Year'].unique()
corn_yield_IA =corn_yield_IA[['County', 'Yield (BU per Acre)']]
corn_yield_IA_avg = corn_yield_IA.groupby(['County'], as_index = False).mean()
corn_yield_IA_avg = corn_yield_IA_avg.sort_values(['Yield (BU per Acre)'], ascending = [0])
corn_yield_IA_avg.head(12)

'''Find the county which has highest avg production in Iowa (the highest corn producing state in the US) for 2010-2018'''
corn_prod_IA = pd.read_csv('M:\Documents\Thesis\Preliminary Analysis\corn_production.csv')
corn_prod_IA = corn_prod_IA[(corn_prod_IA['Year'] >= 2010) & (corn_prod_IA['Year'] <= 2018) & (corn_prod_IA['State'] == 'IOWA')]
corn_prod_IA['County'] = corn_prod_IA['County'].str.title()
corn_prod_IA = corn_prod_IA.rename(columns={'Value': 'Production (BU)'})
corn_prod_IA.head()
corn_prod_IA['Year'].unique()
corn_prod_IA =corn_prod_IA[['County', 'Production (BU)']]
corn_prod_IA_avg = corn_prod_IA.groupby(['County'], as_index = False).mean()
corn_prod_IA_avg['corn_prod_IA_share'] = (corn_prod_IA_avg['Production (BU)'] / corn_prod_IA_avg['Production (BU)'].sum()) * 100
corn_prod_IA_avg = corn_prod_IA_avg.sort_values(['corn_prod_IA_share'], ascending = [0])
corn_prod_IA_avg.head(12)

'''
Iowa has the highest average production share in the country. In terms of average yield also, It figures at the top among the major corn producing states. However in terms of average yield, Washington comes at top but in terms of average production share, it comes at 30th position.

Sioux comes at second position in terms of both average production and average yield. I choose Sioux county as the indicator to the market. Kossuth which has highest average production, comes at 23rd position in terms of average yield for the period. Ida which has the highest average yield, comes at 52nd position in terms of average production.

Choosing a county as an early indicator to the market (weather -> futures price): Iowa - Sioux
'''

'''Average Futures Price for 2010-2018 for months April and onwards'''
price_data = pd.read_csv('M:\Documents\Thesis\Final Data\price_data.csv')
price_data['Date'] = pd.to_datetime(price_data['Date'])
price_data['Year'] = price_data['Date'].dt.year
price_data['Month'] = price_data['Date'].dt.month
price_data = price_data[(price_data['Year'] >= 2010) & (price_data['Year'] <= 2018) & (price_data['Month'] >= 4)]
price_data.tail()
price_data['Year'].unique()
price_data =price_data[['Year', 'Close']]
price_data_avg = price_data.groupby(['Year'], as_index = False).mean()
price_data_avg = price_data_avg.sort_values(['Year'], ascending = [1])
price_data_avg

'''Average Weather Conditions for Sioux-Iowa for 2010-2018 for the corn growing season April - October'''
weather_data = pd.read_csv('M:\Documents\Thesis\Final Data\data_final.csv')
weather_data_1 = weather_data[(weather_data['Year'] >= 2010) & (weather_data['Year'] <= 2018) & (weather_data['Month'] >= 4) & (weather_data['Month'] <= 10) & (weather_data['State'] == 'Iowa') & (weather_data['County'] == 'Sioux')]
weather_data_1.tail()
weather_data_1['Year'].unique()
weather_data_1 = weather_data_1.sort_values(['Date'], ascending = [1])
weather_data_1 =weather_data_1[['Year', 'SurfaceTemperatureFahrenheit_mean']]
weather_data_1.head()

'Sioux - Iowa weather (temperature) impact on futures - Violin plot'
ax1 = sns.violinplot(x = 'Year', y = 'SurfaceTemperatureFahrenheit_mean', data = weather_data_1)
ax1.set_ylabel('Surface Temperature Daily Mean (F)')
ax2 = plt.twinx()
ax2.set_ylabel('Futures Price (Cents/BU)')
sns.lineplot(data = price_data_avg['Close'], color = 'b', ax = ax2)


weather_data_1 = weather_data[(weather_data['Year'] >= 2010) & (weather_data['Year'] <= 2018) & (weather_data['Month'] >= 4) & (weather_data['Month'] <= 10) & (weather_data['State'] == 'Iowa') & (weather_data['County'] == 'Sioux')]
weather_data_1.count()
weather_data_1['Year'].unique()
weather_data_1 = weather_data_1.sort_values(['Date'], ascending = [1])
weather_data_1 =weather_data_1[['Year', 'RelativeHumidity_mean']]
weather_data_1.head()

'Sioux - Iowa weather (temperature) impact on futures - Violin plot'
ax1 = sns.boxplot(x = 'Year', y = 'RelativeHumidity_mean', data = weather_data_1)
ax1.set_ylabel('Relative Humidity Daily Mean (%)')
ax2 = plt.twinx()
ax2.set_ylabel('Futures Price (Cents/BU)')
sns.lineplot(data = price_data_avg['Close'], color = 'b', ax = ax2)


'''Distance between states, Minnesota as benchmark state'''
from math import radians, sin, cos, sqrt, asin
 
def haversine(lat1, lon1, lat2, lon2):
    R = 3961  # Earth radius in miles
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2)**2 + cos(lat1) * cos(lat2) * sin(dLon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

haversine(46.729553, -94.6859, 41.878003, -93.097702) #IA
haversine(46.729553, -94.6859, 40.633125, -89.398528) #IL
haversine(46.729553, -94.6859, 40.551217, -85.602364) #IN
haversine(46.729553, -94.6859, 39.011902, -98.484246) #KS
haversine(46.729553, -94.6859, 44.314844, -85.602364) #MI
haversine(46.729553, -94.6859, 37.964253, -91.831833) #MO
haversine(46.729553, -94.6859, 41.492537, -99.901813) #NE
haversine(46.729553, -94.6859, 40.417287, -82.907123) #OH
haversine(46.729553, -94.6859, 43.969515, -99.901813) #SD


'''Spatial ordering of states with changing benchmark state'''
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


haversine_test = pd.read_excel('M:\Documents\Thesis\Test Data\haversine_test.xlsx', 'Sheet2')
state_x = []
x = ['MN']
while len(haversine_test) > 0:
    print(x)
    state_x.append(x[0])
    haversine_test.loc[:, 'lat1'] = (haversine_test.loc[haversine_test['state'] == x[0]])['latitude'].mean()
    haversine_test.loc[:, 'long1'] = (haversine_test.loc[haversine_test['state'] == x[0]])['longitude'].mean()
    #haversine_test = haversine_test[((haversine_test['latitude'] != haversine_test['lat1']) & (haversine_test['longitude'] != haversine_test['long1']))]
    haversine_test = haversine_test[(haversine_test['state'] != x[0])]
    haversine_test.loc[:, 'distance'] = haversine_np(haversine_test['lat1'], haversine_test['long1'], haversine_test['latitude'], haversine_test['longitude'])
    #x = (np.array((haversine_test.loc[haversine_test['distance'] == haversine_test['distance'].min()])['state'])).tolist()
    x = (haversine_test.loc[haversine_test['distance'] == haversine_test['distance'].min()])['state'].unique()

state_rank = pd.DataFrame(state_x, columns = ['State'])

    





































'''
fig, ax1 = plt.subplots()
ax1.set_ylabel('Surface Temperature Daily Max (F)')
ax1 = sns.violinplot(x = 'Year', y = 'SurfaceTemperatureFahrenheit_max', data = weather_data_1)
#ax1 = sns.barplot(x = weather_data_1['Year'].tolist(), y = weather_data_1['SurfaceTemperatureFahrenheit_max'].tolist())
ax2 = ax1.twinx()
ax2.set_ylabel('Futures Price ($/BU)')
ax2 = sns.lineplot(x = price_data_avg['Year'].tolist(), y = price_data_avg['Close'].tolist())
plt.show()













weather_data_1 =weather_data_1[['Year', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceTemperatureFahrenheit_min']]

variables = ['SurfaceTemperatureFahrenheit_mean']
for x in variables:
    weather_data_1[x+'_year'] = weather_data_1.groupby(['Year'])[x].transform('mean')

variables = ['SurfaceTemperatureFahrenheit_max']
for x in variables:
    weather_data_1[x+'_year'] = weather_data_1.groupby(['Year'])[x].transform('max')

variables = ['SurfaceTemperatureFahrenheit_min']
for x in variables:
    weather_data_1[x+'_year'] = weather_data_1.groupby(['Year'])[x].transform('min')

weather_data_avg = weather_data_1.groupby(['Year'], as_index = False).mean()
weather_data_avg =weather_data_avg[['Year', 'SurfaceTemperatureFahrenheit_mean_year', 'SurfaceTemperatureFahrenheit_max_year', 'SurfaceTemperatureFahrenheit_min_year']]
weather_data_plot = pd.melt(weather_data_avg, id_vars = ['Year'], var_name = 'Metrics', value_name = 'Surface Temperature (F)')
weather_data_plot.head()






