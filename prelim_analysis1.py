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
from math import radians, sin, cos, sqrt, asin


'''Spatial ordering of counties of states with changing benchmark county in the process using Haversine formula'''
weather_current = pd.read_hdf('M:\Documents\Thesis\Final Data\weather_current.h5')
st = {'IA':'Iowa', 'IL':'Illinois', 'IN':'Indiana', 'KS':'Kansas', 'MI':'Michigan', 'MN':'Minnesota', 'MO':'Missouri', 'NE':'Nebraska', 'OH':'Ohio', 'SD':'South Dakota'}
weather_current['State'] = weather_current['state'].map(st)
weather_current = weather_current.rename(columns={'fips': 'stcofips'})
weather_current = weather_current[['State', 'stcofips', 'Latitude', 'Longitude']]
state_county = weather_current.groupby(['State', 'stcofips'], as_index = False).mean()
county_code = pd.read_csv('M:\Documents\Thesis\Final Data\county_code.csv')
state_county_cord = pd.merge(state_county, county_code, how = 'left', on = ['State', 'stcofips'])
state_county_cord.to_csv('M:\Documents\Thesis\Final Data\state_county_cord.csv', index = False)

def haversine(lat1, lon1, lat2, lon2):
    R = 3961  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


state = ['Minnesota', 'South Dakota', 'Nebraska', 'Kansas', 'Missouri', 'Iowa', 'Illinois', 'Indiana', 'Ohio', 'Michigan']
z = [['St. Louis'], ['Roberts'], ['Dakota'], ['Cheyenne'], ['McDonald'], ['Fremont'], ['Jo Daviess'], ['Posey'], ['Hamilton'], ['Berrien']]
i = 0
county_rank = {}
for y in state:
    state_county_cord = pd.read_csv('M:\Documents\Thesis\Final Data\state_county_cord.csv')
    state_county_cord = state_county_cord[(state_county_cord['State'] == y)]
    x = z[i]
    county_x = []
    i = i + 1
    while len(state_county_cord) > 0:
        #print(x)
        county_x.append(x[0])
        state_county_cord.loc[:, 'lat1'] = (state_county_cord.loc[state_county_cord['County'] == x[0]])['Latitude'].mean()
        state_county_cord.loc[:, 'long1'] = (state_county_cord.loc[state_county_cord['County'] == x[0]])['Longitude'].mean()
        state_county_cord = state_county_cord[(state_county_cord['County'] != x[0])]
        state_county_cord.loc[:, 'distance'] = haversine(state_county_cord['lat1'], state_county_cord['long1'], state_county_cord['Latitude'], state_county_cord['Longitude'])
        x = (state_county_cord.loc[state_county_cord['distance'] == state_county_cord['distance'].min()])['County'].unique()

    county_rank[y] = pd.DataFrame(county_x, columns = ['County'])   
    county_rank[y]['State'] = y
    county_rank[y]['State Rank'] = i
    county_rank[y]['Rank'] = np.arange(len(county_rank[y])) + 1
    county_rank[y].head()
    

state_county_rank = pd.DataFrame()
for y in state:
    state_county_rank = state_county_rank.append(county_rank[y])

state_county_rank['County Rank'] = np.arange(len(state_county_rank)) + 1
state_county_rank.to_csv('M:\Documents\Thesis\Final Data\state_county_rank.csv', index = False)


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


'Sioux - Iowa weather (humidity) impact on futures - Box plot'
ax1 = sns.boxplot(x = 'Year', y = 'RelativeHumidity_mean', data = weather_data_1)
ax1.set_ylabel('Relative Humidity Daily Mean (%)')
ax2 = plt.twinx()
ax2.set_ylabel('Futures Price (Cents/BU)')
sns.lineplot(data = price_data_avg['Close'], color = 'b', ax = ax2)























