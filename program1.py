# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:21:52 2019

@author: ssingh17
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''Iowa weather data grouping at daily level'''
Iowa_weather = pd.read_hdf('M:\Thesis\Final Data\Iowa_weather.h5')

Iowa_weather['State'] = 'Iowa'
Iowa_weather.head(10)
Iowa_weather.dtypes

var = ['DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'DownwardSolarRadiation']
for x in var:
    Iowa_weather[x] = Iowa_weather[x].fillna(Iowa_weather[x+'Wsqm'])
var = ['CloudCoverage', 'RelativeHumidity']
for x in var:
    Iowa_weather[x] = Iowa_weather[x].fillna(Iowa_weather[x+'Percent'])
Iowa_weather['MslPressure'] = Iowa_weather['MslPressure'].fillna(Iowa_weather['MslPressureMillibars'])    
Iowa_weather['WindDirection'] = Iowa_weather['WindDirection'].fillna(Iowa_weather['WindDirectionDegrees'])
#print(df11[['DateHrLwt', 'County', 'CloudCoverage', 'CloudCoveragePercent']].loc[df1['CloudCoverage'].isin(['NaN'])])

Iowa_weather = Iowa_weather.drop(['Day', 'Month'], axis = 1) 
Iowa_weather['DateHrLwt'] = pd.to_datetime(Iowa_weather['DateHrLwt'])
Iowa_weather['DateHrCST'] = Iowa_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Iowa_weather = Iowa_weather.sort_values(['DateHrCST'], ascending=[1])
Iowa_weather['Date'] = (Iowa_weather['DateHrCST'].dt.date).astype(str)

#Iowa_weather['Date'] = (Iowa_weather['DateHrLwt'].str.partition(' ')[0]).map(lambda x: x.strip())

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Iowa_weather[x+'_mean'] = Iowa_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Iowa_weather[x+'_max'] = Iowa_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Iowa_weather[x+'_min'] = Iowa_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
   
#Iowa_weather['SurfaceTemperatureFahrenheit_max'] = Iowa_weather.groupby(['State', 'County', 'Date'])['SurfaceTemperatureFahrenheit'].transform('max')
#Iowa_weather['SurfaceTemperatureFahrenheit_min'] = Iowa_weather.groupby(['State', 'County', 'Date'])['SurfaceTemperatureFahrenheit'].transform('min')
    
Iowa_weather = Iowa_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]
Iowa_weather_1 = Iowa_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Iowa_weather_1 = Iowa_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Iowa_weather_1.head()

Iowa_weather_1.to_hdf('M:\Thesis\Final Data\iowa_daily.h5', key = 'Iowa_weather_1')


'''Illinois weather data grouping at daily level'''
Illinois_weather = pd.read_hdf('M:\Thesis\Final Data\Illinois_weather.h5')

Illinois_weather['State'] = 'Illinois'
Illinois_weather.count()

var = ['DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'DownwardSolarRadiation']
for x in var:
    Illinois_weather[x] = Illinois_weather[x].fillna(Illinois_weather[x+'Wsqm'])
var = ['CloudCoverage', 'RelativeHumidity']
for x in var:
    Illinois_weather[x] = Illinois_weather[x].fillna(Illinois_weather[x+'Percent'])
Illinois_weather['MslPressure'] = Illinois_weather['MslPressure'].fillna(Illinois_weather['MslPressureMillibars'])    
Illinois_weather['WindDirection'] = Illinois_weather['WindDirection'].fillna(Illinois_weather['WindDirectionDegrees'])

Illinois_weather = Illinois_weather.drop(['Day', 'Month'], axis = 1) 
Illinois_weather['DateHrLwt'] = pd.to_datetime(Illinois_weather['DateHrLwt'])
Illinois_weather['DateHrCST'] = Illinois_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Illinois_weather = Illinois_weather.sort_values(['DateHrCST'], ascending=[1])
Illinois_weather['Date'] = (Illinois_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Illinois_weather[x+'_mean'] = Illinois_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Illinois_weather[x+'_max'] = Illinois_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Illinois_weather[x+'_min'] = Illinois_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Illinois_weather = Illinois_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]    

Illinois_weather_1 = Illinois_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Illinois_weather_1 = Illinois_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Illinois_weather_1.count()

Illinois_weather_1.to_hdf('M:\Thesis\Final Data\illinois_daily.h5', key = 'Illinois_weather_1')


'''Indiana weather data grouping at daily level'''
Indiana_weather = pd.read_hdf('M:\Thesis\Final Data\Indiana_weather.h5')

Indiana_weather['State'] = 'Indiana'
Indiana_weather.count()

var = ['DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'DownwardSolarRadiation']
for x in var:
    Indiana_weather[x] = Indiana_weather[x].fillna(Indiana_weather[x+'Wsqm'])
var = ['CloudCoverage', 'RelativeHumidity']
for x in var:
    Indiana_weather[x] = Indiana_weather[x].fillna(Indiana_weather[x+'Percent'])
Indiana_weather['MslPressure'] = Indiana_weather['MslPressure'].fillna(Indiana_weather['MslPressureMillibars'])    
Indiana_weather['WindDirection'] = Indiana_weather['WindDirection'].fillna(Indiana_weather['WindDirectionDegrees'])

Indiana_weather = Indiana_weather.drop(['Day', 'Month'], axis = 1) 
Indiana_weather['DateHrLwt'] = pd.to_datetime(Indiana_weather['DateHrLwt'])
Indiana_weather['DateHrCST'] = Indiana_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Indiana_weather = Indiana_weather.sort_values(['DateHrCST'], ascending=[1])
Indiana_weather['Date'] = (Indiana_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Indiana_weather[x+'_mean'] = Indiana_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Indiana_weather[x+'_max'] = Indiana_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Indiana_weather[x+'_min'] = Indiana_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Indiana_weather = Indiana_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]    

Indiana_weather_1 = Indiana_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Indiana_weather_1 = Indiana_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Indiana_weather_1.tail()

Indiana_weather_1.to_hdf('M:\Thesis\Final Data\indiana_daily.h5', key = 'Indiana_weather_1')


'''Kansas weather data grouping at daily level'''
Kansas_weather = pd.read_hdf('M:\Thesis\Final Data\Kansas_weather.h5')

Kansas_weather['State'] = 'Kansas'
Kansas_weather.count()

Kansas_weather = Kansas_weather.rename(columns={'RelativeHumidityPercent': 'RelativeHumidity', 'CloudCoveragePercent': 'CloudCoverage', 'WindDirectionDegrees': 'WindDirection', 'DownwardSolarRadiationWsqm': 'DownwardSolarRadiation', 'DiffuseHorizontalRadiationWsqm': 'DiffuseHorizontalRadiation',  'DirectNormalIrradianceWsqm': 'DirectNormalIrradiance', 'MslPressureMillibars': 'MslPressure'})

Kansas_weather['DateHrLwt'] = pd.to_datetime(Kansas_weather['DateHrLwt'])
Kansas_weather['DateHrCST'] = Kansas_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Kansas_weather = Kansas_weather.sort_values(['DateHrCST'], ascending=[1])
Kansas_weather['Date'] = (Kansas_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Kansas_weather[x+'_mean'] = Kansas_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Kansas_weather[x+'_max'] = Kansas_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Kansas_weather[x+'_min'] = Kansas_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Kansas_weather = Kansas_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]  

Kansas_weather_1 = Kansas_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Kansas_weather_1 = Kansas_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Kansas_weather_1.tail()

Kansas_weather_1.to_hdf('M:\Thesis\Final Data\kansas_daily.h5', key = 'Kansas_weather_1')


'''Michigan weather data grouping at daily level'''
Michigan_weather = pd.read_hdf('M:\Thesis\Final Data\Michigan_weather.h5')

Michigan_weather['State'] = 'Michigan'
Michigan_weather.count()

Michigan_weather = Michigan_weather.rename(columns={'RelativeHumidityPercent': 'RelativeHumidity', 'CloudCoveragePercent': 'CloudCoverage', 'WindDirectionDegrees': 'WindDirection', 'DownwardSolarRadiationWsqm': 'DownwardSolarRadiation', 'DiffuseHorizontalRadiationWsqm': 'DiffuseHorizontalRadiation',  'DirectNormalIrradianceWsqm': 'DirectNormalIrradiance', 'MslPressureMillibars': 'MslPressure'})

Michigan_weather['DateHrLwt'] = pd.to_datetime(Michigan_weather['DateHrLwt'])
Michigan_weather['DateHrCST'] = Michigan_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Michigan_weather = Michigan_weather.sort_values(['DateHrCST'], ascending=[1])
Michigan_weather['Date'] = (Michigan_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Michigan_weather[x+'_mean'] = Michigan_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Michigan_weather[x+'_max'] = Michigan_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Michigan_weather[x+'_min'] = Michigan_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Michigan_weather = Michigan_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]     
    
Michigan_weather_1 = Michigan_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Michigan_weather_1 = Michigan_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Michigan_weather_1.tail()

Michigan_weather_1.to_hdf('M:\Thesis\Final Data\michigan_daily.h5', key = 'Michigan_weather_1')


'''Minnesota weather data grouping at daily level'''
Minnesota_weather = pd.read_hdf('M:\Thesis\Final Data\Minnesota_weather.h5')

Minnesota_weather['State'] = 'Minnesota'
Minnesota_weather.count()

var = ['DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'DownwardSolarRadiation']
for x in var:
    Minnesota_weather[x] = Minnesota_weather[x].fillna(Minnesota_weather[x+'Wsqm'])
var = ['CloudCoverage', 'RelativeHumidity']
for x in var:
    Minnesota_weather[x] = Minnesota_weather[x].fillna(Minnesota_weather[x+'Percent'])
Minnesota_weather['MslPressure'] = Minnesota_weather['MslPressure'].fillna(Minnesota_weather['MslPressureMillibars'])    
Minnesota_weather['WindDirection'] = Minnesota_weather['WindDirection'].fillna(Minnesota_weather['WindDirectionDegrees'])

Minnesota_weather = Minnesota_weather.drop(['Day', 'Month'], axis = 1) 
Minnesota_weather['DateHrLwt'] = pd.to_datetime(Minnesota_weather['DateHrLwt'])
Minnesota_weather['DateHrCST'] = Minnesota_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Minnesota_weather = Minnesota_weather.sort_values(['DateHrCST'], ascending=[1])
Minnesota_weather['Date'] = (Minnesota_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Minnesota_weather[x+'_mean'] = Minnesota_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Minnesota_weather[x+'_max'] = Minnesota_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Minnesota_weather[x+'_min'] = Minnesota_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Minnesota_weather = Minnesota_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]     
    
#Minnesota_weather_1 = Minnesota_weather.groupby(['State', 'County'], as_index = False).mean()
Minnesota_weather_1 = Minnesota_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Minnesota_weather_1 = Minnesota_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Minnesota_weather_1.tail()

Minnesota_weather_1.to_hdf('M:\Thesis\Final Data\minnesota_daily.h5', key = 'Minnesota_weather_1')
Minnesota_weather_1.dtypes

'''Missouri weather data grouping at daily level'''
Missouri_weather = pd.read_hdf('M:\Thesis\Final Data\Missouri_weather.h5')

Missouri_weather['State'] = 'Missouri'
Missouri_weather.count()

Missouri_weather = Missouri_weather.rename(columns={'RelativeHumidityPercent': 'RelativeHumidity', 'CloudCoveragePercent': 'CloudCoverage', 'WindDirectionDegrees': 'WindDirection', 'DownwardSolarRadiationWsqm': 'DownwardSolarRadiation', 'DiffuseHorizontalRadiationWsqm': 'DiffuseHorizontalRadiation',  'DirectNormalIrradianceWsqm': 'DirectNormalIrradiance', 'MslPressureMillibars': 'MslPressure'})

Missouri_weather['DateHrLwt'] = pd.to_datetime(Missouri_weather['DateHrLwt'])
Missouri_weather['DateHrCST'] = Missouri_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Missouri_weather = Missouri_weather.sort_values(['DateHrCST'], ascending=[1])
Missouri_weather['Date'] = (Missouri_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Missouri_weather[x+'_mean'] = Missouri_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Missouri_weather[x+'_max'] = Missouri_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Missouri_weather[x+'_min'] = Missouri_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Missouri_weather = Missouri_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]    
    
Missouri_weather_1 = Missouri_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Missouri_weather_1 = Missouri_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Missouri_weather_1.tail()

Missouri_weather_1.to_hdf('M:\Thesis\Final Data\missouri_daily.h5', key = 'Missouri_weather_1')


'''Nebraska weather data grouping at daily level'''
Nebraska_weather = pd.read_hdf('M:\Thesis\Final Data\Nebraska_weather.h5')

Nebraska_weather['State'] = 'Nebraska'
Nebraska_weather.count()

var = ['DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'DownwardSolarRadiation']
for x in var:
    Nebraska_weather[x] = Nebraska_weather[x].fillna(Nebraska_weather[x+'Wsqm'])
var = ['CloudCoverage', 'RelativeHumidity']
for x in var:
    Nebraska_weather[x] = Nebraska_weather[x].fillna(Nebraska_weather[x+'Percent'])
Nebraska_weather['MslPressure'] = Nebraska_weather['MslPressure'].fillna(Nebraska_weather['MslPressureMillibars'])    
Nebraska_weather['WindDirection'] = Nebraska_weather['WindDirection'].fillna(Nebraska_weather['WindDirectionDegrees'])

Nebraska_weather = Nebraska_weather.drop(['Day', 'Month'], axis = 1) 
Nebraska_weather['DateHrLwt'] = pd.to_datetime(Nebraska_weather['DateHrLwt'])
Nebraska_weather['DateHrCST'] = Nebraska_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Nebraska_weather = Nebraska_weather.sort_values(['DateHrCST'], ascending=[1])
Nebraska_weather['Date'] = (Nebraska_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Nebraska_weather[x+'_mean'] = Nebraska_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Nebraska_weather[x+'_max'] = Nebraska_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Nebraska_weather[x+'_min'] = Nebraska_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Nebraska_weather = Nebraska_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]    
    
Nebraska_weather_1 = Nebraska_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Nebraska_weather_1 = Nebraska_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Nebraska_weather_1.tail()

Nebraska_weather_1.to_hdf('M:\Thesis\Final Data\ nebraska_daily.h5', key = 'Nebraska_weather_1')


'''Ohio weather data grouping at daily level'''
Ohio_weather = pd.read_hdf('M:\Thesis\Final Data\Ohio_weather.h5')

Ohio_weather['State'] = 'Ohio'
Ohio_weather.count()

Ohio_weather = Ohio_weather.rename(columns={'RelativeHumidityPercent': 'RelativeHumidity', 'CloudCoveragePercent': 'CloudCoverage', 'WindDirectionDegrees': 'WindDirection', 'DownwardSolarRadiationWsqm': 'DownwardSolarRadiation', 'DiffuseHorizontalRadiationWsqm': 'DiffuseHorizontalRadiation',  'DirectNormalIrradianceWsqm': 'DirectNormalIrradiance', 'MslPressureMillibars': 'MslPressure'})

Ohio_weather['DateHrLwt'] = pd.to_datetime(Ohio_weather['DateHrLwt'])
Ohio_weather['DateHrCST'] = Ohio_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
Ohio_weather = Ohio_weather.sort_values(['DateHrCST'], ascending=[1])
Ohio_weather['Date'] = (Ohio_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    Ohio_weather[x+'_mean'] = Ohio_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    Ohio_weather[x+'_max'] = Ohio_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    Ohio_weather[x+'_min'] = Ohio_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
Ohio_weather = Ohio_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]      
    
Ohio_weather_1 = Ohio_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
Ohio_weather_1 = Ohio_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
Ohio_weather_1.tail()

Ohio_weather_1.to_hdf('M:\Thesis\Final Data\ohio_daily.h5', key = 'Ohio_weather_1')


'''South Dakota weather data grouping at daily level'''
South_Dakota_weather = pd.read_hdf('M:\Thesis\Final Data\South_Dakota_weather.h5')

South_Dakota_weather['State'] = 'South Dakota'
South_Dakota_weather.count()

South_Dakota_weather = South_Dakota_weather.rename(columns={'RelativeHumidityPercent': 'RelativeHumidity', 'CloudCoveragePercent': 'CloudCoverage', 'WindDirectionDegrees': 'WindDirection', 'DownwardSolarRadiationWsqm': 'DownwardSolarRadiation', 'DiffuseHorizontalRadiationWsqm': 'DiffuseHorizontalRadiation',  'DirectNormalIrradianceWsqm': 'DirectNormalIrradiance', 'MslPressureMillibars': 'MslPressure'})

South_Dakota_weather['DateHrLwt'] = pd.to_datetime(South_Dakota_weather['DateHrLwt'])
South_Dakota_weather['DateHrCST'] = South_Dakota_weather['DateHrLwt'] + datetime.timedelta(hours = 1)
South_Dakota_weather = South_Dakota_weather.sort_values(['DateHrCST'], ascending=[1])
South_Dakota_weather['Date'] = (South_Dakota_weather['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    South_Dakota_weather[x+'_mean'] = South_Dakota_weather.groupby(['State', 'County', 'Date'])[x].transform('mean')
    South_Dakota_weather[x+'_max'] = South_Dakota_weather.groupby(['State', 'County', 'Date'])[x].transform('max')
    South_Dakota_weather[x+'_min'] = South_Dakota_weather.groupby(['State', 'County', 'Date'])[x].transform('min')
    
South_Dakota_weather = South_Dakota_weather[['State', 'County', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]    

South_Dakota_weather_1 = South_Dakota_weather.groupby(['State', 'County', 'Date'], as_index = False).mean()
South_Dakota_weather_1 = South_Dakota_weather_1.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
South_Dakota_weather_1.tail()

South_Dakota_weather_1.to_hdf('M:\Thesis\Final Data\south_dakota_daily.h5', key = 'South_Dakota_weather_1')


'''Append corn belt daily weather data'''
var = ['iowa', 'illinois', 'indiana', 'kansas', 'michigan', 'minnesota', 'missouri', ' nebraska', 'ohio', 'south_dakota']
df_weather = pd.DataFrame()
for x in var:
    for f in glob.glob('M:/Thesis/Final Data/' + x + '_daily' + '.h5'):
        df2 = pd.read_hdf(f)
        df_weather = df_weather.append(df2, sort = False, ignore_index=True)
df_weather.count()

'''Merging County code files'''
df_ccode = pd.read_csv('M:\Thesis\county_cb_fips.csv')
st = {'IA':'Iowa', 'IL':'Illinois', 'IN':'Indiana', 'KS':'Kansas', 'MI':'Michigan', 'MN':'Minnesota', 'MO':'Missouri', 'NE':'Nebraska', 'OH':'Ohio', 'SD':'South Dakota'}
df_ccode['State'] = df_ccode['state'].map(st)
df_ccode = df_ccode.rename(columns={'county': 'County', 'fips': 'stcofips'})
df_ccode = df_ccode[['State', 'County', 'stcofips']]
df_ccode.loc[(df_ccode['State'] == 'Missouri') & (df_ccode['County'] == 'Mcdonald'), 'County'] = 'McDonald'
df_ccode.loc[(df_ccode['State'] == 'Nebraska') & (df_ccode['County'] == 'McPherson'), 'County'] = 'Mcpherson'

counties_leftout = [pd.Series(['South Dakota', 'Shannon', 46113], index = df_ccode.columns), pd.Series(['Minnesota', 'Cook', 27031], index = df_ccode.columns), pd.Series(['Minnesota', 'Lake', 27075], index = df_ccode.columns), pd.Series(['Minnesota', 'Ramsey', 27123], index = df_ccode.columns), pd.Series(['Missouri', 'St. Louis City', 29510], index = df_ccode.columns)]
df_ccode = df_ccode.append(counties_leftout, ignore_index=True)
df_ccode = df_ccode.sort_values(['State', 'County', 'stcofips'], ascending=[1, 1, 1])
df_weather_f = pd.merge(df_weather, df_ccode, how = 'left', on = ['State', 'County'])
#df_weather_f[df_weather_f['stcofips'].isnull()]['State'].unique()
df_weather_f.count()

'''Appending current (post 2016) weather data'''
weather_current = pd.read_hdf('M:\Thesis\Final Data\weather_current.h5')
weather_current.tail()

st = {'IA':'Iowa', 'IL':'Illinois', 'IN':'Indiana', 'KS':'Kansas', 'MI':'Michigan', 'MN':'Minnesota', 'MO':'Missouri', 'NE':'Nebraska', 'OH':'Ohio', 'SD':'South Dakota'}
weather_current['State'] = weather_current['state'].map(st)
#weather_current = weather_current[weather_current['State'] != 'Ohio']

weather_current = weather_current.rename(columns={'RelativeHumidityPercent': 'RelativeHumidity', 'CloudCoveragePercent': 'CloudCoverage', 'WindDirectionDegrees': 'WindDirection', 'DownwardSolarRadiationWsqm': 'DownwardSolarRadiation', 'DiffuseHorizontalRadiationWsqm': 'DiffuseHorizontalRadiation',  'DirectNormalIrradianceWsqm': 'DirectNormalIrradiance', 'MslPressureMillibars': 'MslPressure', 'fips': 'stcofips'})

weather_current['DateHrLwt'] = pd.to_datetime(weather_current['DateHrLwt'])
weather_current['DateHrCST'] = weather_current['DateHrLwt'] + datetime.timedelta(hours = 1)
weather_current = weather_current.sort_values(['DateHrCST'], ascending=[1])
weather_current['Date'] = (weather_current['DateHrCST'].dt.date).astype(str)

variables = ['SurfaceTemperatureFahrenheit', 'SurfaceDewpointTemperatureFahrenheit', 'SurfaceWetBulbTemperatureFahrenheit', 'RelativeHumidity', 'SurfaceAirPressureMillibars', 'CloudCoverage', 'WindChillTemperatureFahrenheit', 'ApparentTemperatureFahrenheit', 'WindSpeedMph', 'WindDirection', 'PrecipitationPreviousHourInches', 'DownwardSolarRadiation', 'DiffuseHorizontalRadiation', 'DirectNormalIrradiance', 'MslPressure', 'HeatIndexFahrenheit', 'SnowfallInches', 'SurfaceWindGustsMph']
for x in variables:
    weather_current[x+'_mean'] = weather_current.groupby(['State', 'stcofips', 'Date'])[x].transform('mean')
    weather_current[x+'_max'] = weather_current.groupby(['State', 'stcofips', 'Date'])[x].transform('max')
    weather_current[x+'_min'] = weather_current.groupby(['State', 'stcofips', 'Date'])[x].transform('min')
    
weather_current = weather_current[['State', 'stcofips', 'Date', 'SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max', 'SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']]      
    
weather_current_1 = weather_current.groupby(['State', 'stcofips', 'Date'], as_index = False).mean()
#df_ccode = df_ccode[df_ccode['State'] != 'Ohio']

weather_current_f = pd.merge(weather_current_1, df_ccode, how = 'left', on = ['State', 'stcofips'])
df_weather_final = df_weather_f.append(weather_current_f, sort = False, ignore_index=True)
df_weather_final = df_weather_final.sort_values(['Date', 'State', 'County'], ascending=[1, 1, 1])
df_weather_final.count()
#weather_current_f[weather_current_f['State'] == 'South Dakota']['County'].unique()

'''Combining Soil Data '''
df_soil = pd.read_excel('M:\Thesis\Weather Data\soil.xlsx')
df_soil.columns = df_soil.columns.str.replace(' ', '')
#df_soil['State'] = df_soil['State'].replace({'South Dakota': 'South_Dakota'})
df_soil = df_soil[['State', 'stcofips', 'ffd', 'sandtotal', 'silttotal', 'claytotal', 'om', 'bulkDensity', 'lep', 'caco3', 'ec', 'soc0_150', 'rootznaws', 'droughty', 'sand', 'share_cropland']]

counties_leftout = [('Minnesota', 27031), ('Minnesota', 27075), ('Minnesota', 27123), ('Missouri', 29510), ('Michigan', 26039), ('Michigan', 26053), ('Michigan', 26083), ('Michigan', 26103), ('Michigan', 26143), ('Ohio', 39035)]
df_part1 = pd.DataFrame(counties_leftout, columns = ['State' , 'stcofips']) 
df_part2 = pd.DataFrame(columns=['ffd', 'sandtotal', 'silttotal', 'claytotal', 'om', 'bulkDensity', 'lep', 'caco3', 'ec', 'soc0_150', 'rootznaws', 'droughty', 'sand', 'share_cropland'])
df_part2 = df_part2.astype(dtype = {'ffd': float, 'sandtotal': float, 'silttotal': float, 'claytotal': float, 'om': float, 'bulkDensity': float, 'lep': float, 'caco3': float, 'ec': float, 'soc0_150': float, 'rootznaws': float, 'droughty': float, 'sand': float, 'share_cropland': float})
df_s = df_part1.append(df_part2, sort = False, ignore_index=True)
df_s['stcofips'] = df_s['stcofips'].astype(int)
df_soil = df_soil.append(df_s, sort = False, ignore_index=True)
var = ['ffd', 'sandtotal', 'silttotal', 'claytotal', 'om', 'bulkDensity', 'lep', 'caco3', 'ec', 'soc0_150', 'rootznaws', 'droughty', 'sand', 'share_cropland']
for y in var:
    df_soil[y] = df_soil.groupby(['State'])[y].apply(lambda x: x.fillna(x.mean()))
df_soil = df_soil.sort_values(['State', 'stcofips'], ascending = [1, 1])    
df_final = pd.merge(df_weather_final, df_soil, how = 'left', on = ['State', 'stcofips'])
#df_final[df_final['ffd'].isnull()]['State'].unique()
#df_final[(df_final['ffd'].isnull()) & (df_final['State'] == 'Ohio')]['County'].unique()
df_final['Year'] = (df_final['Date'].str.partition('-')[0]).map(lambda x: x.strip()).astype(int)
df_final['Day'] = (df_final['Date'].str.rpartition('-')[2]).map(lambda x: x.strip()).astype(int)
df_final['Month'] = ((df_final['Date'].str.partition('-')[2]).str.partition('-')[0]).map(lambda x: x.strip()).astype(int)
df_final['Date'] = pd.to_datetime(df_final['Date'])
df_final = df_final.sort_values(['State', 'stcofips', 'Date'], ascending = [1, 1, 1])
df_final.count()
#len(df_final[(df_final['Year'] == 1996) & (df_final['State'] == 'Indiana')]['County'].unique())

'''Creating variable PDSI'''
df_pdsi_table = pd.read_csv('M:\Thesis\PDSI Data\climdiv-pdsidv-v1.0.0-20191104.txt', delim_whitespace = True, names = ('ID', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'), dtype = {'ID': str})

df_pdsi=pd.melt(df_pdsi_table,id_vars=['ID'],var_name='Month', value_name='PDSI')
df_pdsi['Month'] = df_pdsi['Month'].astype(int)
df_pdsi = df_pdsi.sort_values(['ID', 'Month'], ascending = [1, 1]).reset_index(drop = True)
df_pdsi['State_code'] = df_pdsi['ID'].map(lambda x: x[0:2])
df_pdsi['Year'] = (df_pdsi['ID'].map(lambda x: x[6:10])).astype(int)
df_pdsi['CD'] = (df_pdsi['ID'].map(lambda x: x[0:4])).astype(int)

df_pdsi = df_pdsi[(df_pdsi['State_code'] == '11') | (df_pdsi['State_code'] == '12') | (df_pdsi['State_code'] == '13') | (df_pdsi['State_code'] == '14') | (df_pdsi['State_code'] == '20') | (df_pdsi['State_code'] == '21') | (df_pdsi['State_code'] == '23') | (df_pdsi['State_code'] == '25') | (df_pdsi['State_code'] == '33') | (df_pdsi['State_code'] == '39')]
st_code = {'11': 'Illinois', '12': 'Indiana', '13': 'Iowa', '14': 'Kansas', '20': 'Michigan', '21': 'Minnesota', '23': 'Missouri', '25': 'Nebraska', '33': 'Ohio', '39': 'South Dakota'}
df_pdsi['State'] = df_pdsi['State_code'].map(st_code)

df_pdsi_code = pd.read_csv('M:\Thesis\PDSI Data\SpatialJoin_CDs_to_Counties_Final.txt', sep = ',', header = 0, dtype = {'CD': str})
df_pdsi_code = df_pdsi_code.rename(columns={'FIPS': 'stcofips'})
df_pdsi_code['State_code'] = df_pdsi_code['CD'].map(lambda x: x[0:-2])
df_pdsi_code = df_pdsi_code[(df_pdsi_code['State_code'] == '11') | (df_pdsi_code['State_code'] == '12') | (df_pdsi_code['State_code'] == '13') | (df_pdsi_code['State_code'] == '14') | (df_pdsi_code['State_code'] == '20') | (df_pdsi_code['State_code'] == '21') | (df_pdsi_code['State_code'] == '23') | (df_pdsi_code['State_code'] == '25') | (df_pdsi_code['State_code'] == '33') | (df_pdsi_code['State_code'] == '39')]

df_pdsi_code = df_pdsi_code.sort_values(['stcofips', 'State_code', 'CD'], ascending = [1, 1, 1])
df_pdsi_code_dup = df_pdsi_code[df_pdsi_code.duplicated(['stcofips'])]
df_pdsi_code_f = df_pdsi_code.drop_duplicates(subset = ['stcofips'], keep = 'first') 
df_pdsi_code_f['CD'] = df_pdsi_code_f['CD'].astype(int)

df_pdsi_f = pd.merge(df_pdsi, df_pdsi_code_f, how = 'inner', on = ['CD'])
df_pdsi_f = df_pdsi_f.sort_values(['State', 'stcofips', 'Year', 'Month'], ascending = [1, 1, 1, 1])
df_pdsi_f = df_pdsi_f[['State', 'stcofips', 'Year', 'Month', 'PDSI']]
df_f = pd.merge(df_final, df_pdsi_f, how = 'left', on = ['State', 'stcofips', 'Year', 'Month'])
#df_f[df_f['PDSI'].isnull()]['State'].unique()

'''Creating variable GDD'''
df_f['GDD'] = (df_f['SurfaceTemperatureFahrenheit_max'].apply(lambda x: max(min(86, x), 50)) + df_f['SurfaceTemperatureFahrenheit_min'].apply(lambda x: max(50, x)) - 2*50) / 2
#df_f = df_f[(df_f['Month'] > 3) & (df_f['Month'] < 11)]
df_f = df_f.sort_values(['Date', 'State', 'County'], ascending = [1, 1, 1])
df_f['Day_of_week'] = df_f['Date'].dt.day_name()
df_f['Week_of_year'] = df_f['Date'].dt.week
df_f['Date_week_seq'] = 100*df_f['Year'] + df_f['Week_of_year']
df_f.count()


'''Merging Corn Futures Prices'''
df_price = pd.DataFrame()
for f in glob.glob('M:\Thesis\Corn Futures Prices\Dec_Corn_Futures\*.csv'):
   df = pd.read_csv(f, skiprows = [0], skipfooter = 1, engine = 'python')
   f_str = str(f)
   abc = (f_str.split('ZCZ', 1)[1]).split('_Barchart', 1)[0]
   if abc > str(79):
       year = '19' + abc
   else:
       year = '20' + abc 
   df['Contract Year'] = year
   df['Business Year'] = (df['Date Time'].str.partition('-')[0]).map(lambda x: x.strip())
   df = df[(df['Contract Year'] == df['Business Year'])]
   df_price = df_price.append(df, ignore_index=True)

df_price = df_price.rename(columns={'Date Time': 'Date'})
df_price['Date'] = pd.to_datetime(df_price['Date'])
df_price['Business Year'] = df_price['Business Year'].astype(int)
df_price = df_price.sort_values(['Date'], ascending = [1])
df_price['Day_of_week'] = df_price['Date'].dt.day_name()
df_price['Week_of_year'] = df_price['Date'].dt.week
df_price['Open_first'] = df_price.groupby(['Business Year', 'Week_of_year'])['Open'].transform('first')
df_price['Close_last'] = df_price.groupby(['Business Year', 'Week_of_year'])['Close'].transform('last')
df_price['Date_week_seq'] = 100*df_price['Business Year'] + df_price['Week_of_year']
#df_price['Close_next'] = df_price['Close'].shift(periods = -1)
#df_price['Close_prev_4'] = df_price['Close'].shift(periods = 4)
df_price['Close_prev'] = df_price['Close'].shift(periods = 1)
df_price['Close_pcch_cc'] = ((df_price['Close'] / df_price['Close_prev']) - 1)*100
df_price['Close_pcch_co'] = ((df_price['Close'] / df_price['Open']) - 1)*100
df_price['Price_pcch_wco'] = ((df_price['Close_last'] / df_price['Open_first']) - 1)*100
#df_price['Close_pcch_next'] = df_price['Close_pcch'].shift(periods = -1)
df_price1 = df_price[['Date', 'Open', 'Open_first', 'High', 'Low', 'Close', 'Close_last', 'Close_pcch_cc', 'Close_pcch_co', 'Price_pcch_wco']]
df_price1.to_csv('M:\Thesis\Final Data\price_data.csv', index = False)

dff = pd.merge(df_f, df_price1, how = 'left', on = ['Date'])
dff['Date_seq'] = 10000*dff['Year'] + 100*dff['Month'] + 1*dff['Day']
dff = dff.sort_values(['Date_seq', 'State', 'County'], ascending = [1, 1, 1])
dff.count()


'''Droping 5 counties which are not available in post 2016 data'''
'''
dfff = dff[(dff['stcofips'] != 27031) & (dff['stcofips'] != 27075) & (dff['stcofips'] != 27123) & (dff['stcofips'] != 29510) & (dff['stcofips'] != 46113)]   
dfff.count()

dfff = dfff.sort_values(['Date_seq', 'Date', 'State', 'County'], ascending = [1, 1, 1, 1])
dfff.to_hdf('M:\Thesis\Final Data\data_final.h5', key = 'dfff')
dfff.to_csv('M:\Thesis\Final Data\data_final.csv', index = False)
'''

'''Adding county map order for states to incorporate local spatial characteristics in CNN model''' 
df_county_total_map_order = pd.read_csv('M:\Thesis\Final Data\county_total_map_order.csv')
df_county_total_map_order.count()
dfff = pd.merge(dff, df_county_total_map_order, how = 'inner', on = ['State', 'County', 'stcofips'])
dfff = dfff.sort_values(['Date_seq', 'State', 'County'], ascending = [1, 1, 1])
dfff.count()


'''Droping counties, which have no yield data for certain years, more than 2 years in this case (No corn production) from the master file'''
county_keep = pd.read_csv('M:\Thesis\Final Data\county_keep.csv')
#county_keep = pd.read_csv('M:\Thesis\Final Data\county_high_yield.csv')
county_keep.loc[(county_keep['State'] == 'Illinois') & (county_keep['County'] == 'St Clair'), 'County'] = 'St. Clair'
county_keep.loc[(county_keep['State'] == 'Illinois') & (county_keep['County'] == 'Mclean'), 'County'] = 'McLean'
county_keep.loc[(county_keep['State'] == 'Illinois') & (county_keep['County'] == 'Mchenry'), 'County'] = 'McHenry'
county_keep.loc[(county_keep['State'] == 'Illinois') & (county_keep['County'] == 'Mcdonough'), 'County'] = 'McDonough'
county_keep.loc[(county_keep['State'] == 'Illinois') & (county_keep['County'] == 'De Kalb'), 'County'] = 'DeKalb'
county_keep.loc[(county_keep['State'] == 'Indiana') & (county_keep['County'] == 'St. Joseph'), 'County'] = 'St Joseph'
county_keep.loc[(county_keep['State'] == 'Kansas') & (county_keep['County'] == 'Mcpherson'), 'County'] = 'McPherson'
county_keep.loc[(county_keep['State'] == 'Missouri') & (county_keep['County'] == 'St Louis'), 'County'] = 'St. Louis'
county_keep.loc[(county_keep['State'] == 'Missouri') & (county_keep['County'] == 'St Charles'), 'County'] = 'St. Charles'
county_keep.loc[(county_keep['State'] == 'South Dakota') & (county_keep['County'] == 'Mcpherson'), 'County'] = 'McPherson'
county_keep.loc[(county_keep['State'] == 'South Dakota') & (county_keep['County'] == 'Mccook'), 'County'] = 'McCook'
county_keep['count'] = 1
dffff = pd.merge(dfff, county_keep, how = 'inner', on = ['State', 'County'])
#dffff[(dffff['count'].isnull()) & (dffff['State'] == 'Ohio')]['County'].unique()
#dffff =dffff[dffff['count'].isnull()]
#dfff = dfff.sort_values(['Date_seq', 'Date', 'State', 'County'], ascending = [1, 1, 1, 1])
#dffff['Date'] = pd.to_datetime(dffff['Date'])
dffff = dffff.sort_values(['Date_seq', 'Date', 'State Rank', 'County Rank'], ascending = [1, 1, 1, 1])
dffff.dtypes
dffff.head()
dffff.to_hdf('M:\Thesis\Final Data\data_final.h5', key = 'dffff')
dffff.to_csv('M:\Thesis\Final Data\data_final.csv', index = False)


'''Creating data to measure weekend weather impact'''
data_weekend = dffff.copy() #created with left merge in '''Merging Corn Futures Prices''' section to keep weekend
data_weekend.count()                                                                                                                                                  
data_weekend = data_weekend[(data_weekend['Day_of_week'] == 'Friday') | (data_weekend['Day_of_week'] == 'Saturday') | (data_weekend['Day_of_week'] == 'Sunday')]

variables = ['SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean']
for x in variables:
    data_weekend[x+'_wkmean'] = data_weekend.groupby(['State', 'stcofips', 'Year', 'Date_week_seq'])[x].transform('mean')

variables = ['SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max']
for x in variables:
    data_weekend[x+'_wkmax'] = data_weekend.groupby(['State', 'stcofips', 'Year', 'Date_week_seq'])[x].transform('max')

variables = ['SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']
for x in variables:
    data_weekend[x+'_wkmin'] = data_weekend.groupby(['State', 'stcofips', 'Year', 'Date_week_seq'])[x].transform('min')

data_weekend_1 = data_weekend[['State', 'County', 'stcofips', 'State Rank', 'County Rank', 'Year', 'Date_week_seq', 'Week_of_year', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'SurfaceDewpointTemperatureFahrenheit_mean_wkmean', 'SurfaceWetBulbTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'CloudCoverage_mean_wkmean', 'WindChillTemperatureFahrenheit_mean_wkmean', 'ApparentTemperatureFahrenheit_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'DownwardSolarRadiation_mean_wkmean', 'DiffuseHorizontalRadiation_mean_wkmean', 'DirectNormalIrradiance_mean_wkmean', 'MslPressure_mean_wkmean', 'HeatIndexFahrenheit_mean_wkmean', 'SnowfallInches_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'SurfaceTemperatureFahrenheit_max_wkmax', 'SurfaceDewpointTemperatureFahrenheit_max_wkmax', 'SurfaceWetBulbTemperatureFahrenheit_max_wkmax', 'RelativeHumidity_max_wkmax', 'SurfaceAirPressureMillibars_max_wkmax', 'CloudCoverage_max_wkmax', 'WindChillTemperatureFahrenheit_max_wkmax', 'ApparentTemperatureFahrenheit_max_wkmax', 'WindSpeedMph_max_wkmax', 'WindDirection_max_wkmax', 'PrecipitationPreviousHourInches_max_wkmax', 'DownwardSolarRadiation_max_wkmax', 'DiffuseHorizontalRadiation_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'MslPressure_max_wkmax', 'HeatIndexFahrenheit_max_wkmax', 'SnowfallInches_max_wkmax', 'SurfaceWindGustsMph_max_wkmax', 'SurfaceTemperatureFahrenheit_min_wkmin', 'SurfaceDewpointTemperatureFahrenheit_min_wkmin', 'SurfaceWetBulbTemperatureFahrenheit_min_wkmin', 'RelativeHumidity_min_wkmin', 'SurfaceAirPressureMillibars_min_wkmin', 'CloudCoverage_min_wkmin', 'WindChillTemperatureFahrenheit_min_wkmin', 'ApparentTemperatureFahrenheit_min_wkmin', 'WindSpeedMph_min_wkmin', 'WindDirection_min_wkmin', 'PrecipitationPreviousHourInches_min_wkmin', 'DownwardSolarRadiation_min_wkmin', 'DiffuseHorizontalRadiation_min_wkmin', 'DirectNormalIrradiance_min_wkmin', 'MslPressure_min_wkmin', 'HeatIndexFahrenheit_min_wkmin', 'SnowfallInches_min_wkmin', 'SurfaceWindGustsMph_min_wkmin']]
data_weekend_2 = data_weekend_1.groupby(['State', 'County', 'stcofips', 'Year', 'Date_week_seq'], as_index = False).mean()

df_price2 = df_price.copy()
df_price2 = df_price2.sort_values(['Date_week_seq', 'Date'], ascending = [1, 1])
df_price2['Open_wk_first_next'] = np.where(df_price2['Date_week_seq'] == df_price2['Date_week_seq'].shift(-1), np.nan, df_price2['Open_first'].shift(-1))
#df_price2['J1'] = df_price2['Open_first'].shift(-1)
#df_price2['Open_wk_first_next'] = np.where(df_price2['J1'] == df_price2['Open_first'], np.nan, df_price2['J1'])
df_price2['Open_wk_first_next'] = df_price2.groupby(['Date_week_seq', 'Week_of_year'])['Open_wk_first_next'].transform('mean')
df_price2['Close_wk_last'] = df_price2['Close_last']
df_price2['Weekend_pcch'] = ((df_price2['Open_wk_first_next'] / df_price2['Close_wk_last']) - 1)*100
#df_price2.to_csv('M:\Thesis\Test Data\price_data.csv', index = False)

df_price3 = df_price2[['Date_week_seq', 'Close_wk_last', 'Open_wk_first_next', 'Weekend_pcch']]
df_price3 = df_price3.groupby(['Date_week_seq'], as_index = False).mean()
df_price3.tail()

data_final_weekend = pd.merge(data_weekend_2, df_price3, how = 'inner', on = ['Date_week_seq'])
data_final_weekend = data_final_weekend.sort_values(['Year', 'Date_week_seq', 'State Rank', 'County Rank'], ascending = [1, 1, 1, 1])
data_final_weekend.to_csv('M:\Thesis\Final Data\data_final_weekend.csv', index = False)
data_final_weekend.count()


'''Creating data at week level'''
df_week = df_f.sort_values(['State', 'County', 'Date'], ascending = [1, 1, 1])
#df_week = df_week[(df_week['Month'] > 3) & (df_week['Month'] < 12)]
df_week['Day_of_week'] = df_f['Date'].dt.day_name()
df_week['Week_of_year'] = df_week['Date'].dt.week
df_week['Date_week_seq'] = 100*df_week['Year'] + df_week['Week_of_year']

variables = ['SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean']
for x in variables:
    df_week[x+'_wkmean'] = df_week.groupby(['State', 'stcofips', 'Year', 'Date_week_seq'])[x].transform('mean')

variables = ['SurfaceTemperatureFahrenheit_max', 'SurfaceDewpointTemperatureFahrenheit_max', 'SurfaceWetBulbTemperatureFahrenheit_max', 'RelativeHumidity_max', 'SurfaceAirPressureMillibars_max', 'CloudCoverage_max', 'WindChillTemperatureFahrenheit_max', 'ApparentTemperatureFahrenheit_max', 'WindSpeedMph_max', 'WindDirection_max', 'PrecipitationPreviousHourInches_max', 'DownwardSolarRadiation_max', 'DiffuseHorizontalRadiation_max', 'DirectNormalIrradiance_max', 'MslPressure_max', 'HeatIndexFahrenheit_max', 'SnowfallInches_max', 'SurfaceWindGustsMph_max']
for x in variables:
    df_week[x+'_wkmax'] = df_week.groupby(['State', 'stcofips', 'Year', 'Date_week_seq'])[x].transform('max')

variables = ['SurfaceTemperatureFahrenheit_min', 'SurfaceDewpointTemperatureFahrenheit_min', 'SurfaceWetBulbTemperatureFahrenheit_min', 'RelativeHumidity_min', 'SurfaceAirPressureMillibars_min', 'CloudCoverage_min', 'WindChillTemperatureFahrenheit_min', 'ApparentTemperatureFahrenheit_min', 'WindSpeedMph_min', 'WindDirection_min', 'PrecipitationPreviousHourInches_min', 'DownwardSolarRadiation_min', 'DiffuseHorizontalRadiation_min', 'DirectNormalIrradiance_min', 'MslPressure_min', 'HeatIndexFahrenheit_min', 'SnowfallInches_min', 'SurfaceWindGustsMph_min']
for x in variables:
    df_week[x+'_wkmin'] = df_week.groupby(['State', 'stcofips', 'Year', 'Date_week_seq'])[x].transform('min')

df_week = df_week[['State', 'County', 'stcofips', 'Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'SurfaceDewpointTemperatureFahrenheit_mean_wkmean', 'SurfaceWetBulbTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'CloudCoverage_mean_wkmean', 'WindChillTemperatureFahrenheit_mean_wkmean', 'ApparentTemperatureFahrenheit_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'DownwardSolarRadiation_mean_wkmean', 'DiffuseHorizontalRadiation_mean_wkmean', 'DirectNormalIrradiance_mean_wkmean', 'MslPressure_mean_wkmean', 'HeatIndexFahrenheit_mean_wkmean', 'SnowfallInches_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'SurfaceTemperatureFahrenheit_max_wkmax', 'SurfaceDewpointTemperatureFahrenheit_max_wkmax', 'SurfaceWetBulbTemperatureFahrenheit_max_wkmax', 'RelativeHumidity_max_wkmax', 'SurfaceAirPressureMillibars_max_wkmax', 'CloudCoverage_max_wkmax', 'WindChillTemperatureFahrenheit_max_wkmax', 'ApparentTemperatureFahrenheit_max_wkmax', 'WindSpeedMph_max_wkmax', 'WindDirection_max_wkmax', 'PrecipitationPreviousHourInches_max_wkmax', 'DownwardSolarRadiation_max_wkmax', 'DiffuseHorizontalRadiation_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'MslPressure_max_wkmax', 'HeatIndexFahrenheit_max_wkmax', 'SnowfallInches_max_wkmax', 'SurfaceWindGustsMph_max_wkmax', 'SurfaceTemperatureFahrenheit_min_wkmin', 'SurfaceDewpointTemperatureFahrenheit_min_wkmin', 'SurfaceWetBulbTemperatureFahrenheit_min_wkmin', 'RelativeHumidity_min_wkmin', 'SurfaceAirPressureMillibars_min_wkmin', 'CloudCoverage_min_wkmin', 'WindChillTemperatureFahrenheit_min_wkmin', 'ApparentTemperatureFahrenheit_min_wkmin', 'WindSpeedMph_min_wkmin', 'WindDirection_min_wkmin', 'PrecipitationPreviousHourInches_min_wkmin', 'DownwardSolarRadiation_min_wkmin', 'DiffuseHorizontalRadiation_min_wkmin', 'DirectNormalIrradiance_min_wkmin', 'MslPressure_min_wkmin', 'HeatIndexFahrenheit_min_wkmin', 'SnowfallInches_min_wkmin', 'SurfaceWindGustsMph_min_wkmin']]
df_week_1 = df_week.groupby(['State', 'County', 'stcofips', 'Year', 'Date_week_seq'], as_index = False).mean()

df_price_2 = df_price[['Date_week_seq', 'Open_first', 'Close_last', 'Price_pcch_wco']]
df_price_1 = df_price_2.groupby(['Date_week_seq'], as_index = False).mean()

df_wp = pd.merge(df_week_1, df_price_1, how = 'inner', on = ['Date_week_seq'])
df_wp1 = pd.merge(df_wp, df_county_total_map_order, how = 'inner', on = ['State', 'County', 'stcofips'])
df_wp2 = pd.merge(df_wp1, county_keep, how = 'inner', on = ['State', 'County'])
df_wp2 = df_wp2.sort_values(['Year', 'Date_week_seq', 'State Rank', 'County Rank'], ascending = [1, 1, 1, 1])
df_wp2.to_csv('M:\Thesis\Final Data\data_final_week.csv', index = False)

#df_wp2.count()
#df_wp2.dtypes
#print(df_wp[df_wp['Price_pcch_wco'].isnull()])
#df_wp[df_wp['Price_pcch_wco'].isnull()]['Date_week_seq'].unique()
#df_wp2[(df_wp2['count'].isnull()) & (df_wp2['State'] == 'Iowa')]['County'].unique()

'''Collapsing data at State level instead of County level'''
dfa = pd.read_csv('M:\Thesis\Final Data\data_final.csv')
dfa['Date'] = pd.to_datetime(dfa['Date']) 
dfb = dfa.groupby(['Date', 'Date_seq', 'State'], as_index = False).max()
dfb = dfb.sort_values(['Date', 'Date_seq', 'State Rank'], ascending = [1, 1, 1])
dfb.dtypes
dfb.tail()
dfb.to_csv('M:\Thesis\Final Data\data_final_st.csv', index = False)


'''Create unanticipated weather variables'''
dfa = pd.read_csv('M:\Thesis\Final Data\data_final.csv')
dfa = dfa[(dfa['Year'] > 1980)]
dfa =dfa[['State', 'County', 'stcofips', 'Month', 'Year', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DiffuseHorizontalRadiation_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'PDSI', 'GDD']]
dfb = dfa.groupby(['State', 'County', 'stcofips', 'Year', 'Month'], as_index = False).mean()
dfb = dfb.sort_values(['State', 'County', 'stcofips', 'Month', 'Year'], ascending=[1, 1, 1, 1, 1])
variables = ['SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DiffuseHorizontalRadiation_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'PDSI', 'GDD']

'Rolling window = 2, i.e. averaging last two years weather for that month to crate a variable callled as anticipated weather variable'
for aa in variables:
    dfb[aa + '_mmean'] = dfb.groupby(['State', 'County', 'stcofips', 'Month'])[aa].apply(lambda x: x.rolling(window = 2).mean())
    dfb[aa + '_mmean'] = dfb.groupby(['State', 'County', 'stcofips', 'Month'])[aa + '_mmean'].shift(1)
dfb =dfb[['State', 'County', 'stcofips', 'Month', 'Year', 'SurfaceTemperatureFahrenheit_mean_mmean', 'RelativeHumidity_mean_mmean', 'SurfaceAirPressureMillibars_mean_mmean', 'CloudCoverage_mean_mmean', 'WindSpeedMph_mean_mmean', 'WindDirection_mean_mmean', 'PrecipitationPreviousHourInches_mean_mmean', 'DiffuseHorizontalRadiation_mean_mmean', 'MslPressure_mean_mmean', 'SnowfallInches_mean_mmean', 'SurfaceWindGustsMph_mean_mmean', 'PDSI_mmean', 'GDD_mmean']]
dfb = dfb[(dfb['SurfaceTemperatureFahrenheit_mean_mmean'].notnull()) & (dfb['RelativeHumidity_mean_mmean'].notnull()) & (dfb['GDD_mmean'].notnull())]

dfc = pd.read_csv('M:\Thesis\Final Data\data_final.csv')
dfd = pd.merge(dfc, dfb, how = 'inner', on = ['State', 'County', 'stcofips', 'Year', 'Month'])
#dfd = dfd.sort_values(['Year', 'Month', 'Day', 'Date_seq', 'Date', 'State Rank', 'County Rank'], ascending = [1, 1, 1, 1, 1, 1, 1])

variables = ['SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DiffuseHorizontalRadiation_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'PDSI', 'GDD']
for ab in variables:
    dfd[ab + '_u'] = dfd[ab] - dfd[ab + '_mmean']

dfd['Date'] = pd.to_datetime(dfd['Date'])    
dfd = dfd.sort_values(['Date_seq', 'Date', 'State Rank', 'County Rank'], ascending = [1, 1, 1, 1])
dfe = dfd[['Year', 'Date_seq', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_u', 'RelativeHumidity_mean_u', 'SurfaceAirPressureMillibars_mean_u', 'CloudCoverage_mean_u', 'WindSpeedMph_mean_u', 'WindDirection_mean_u', 'PrecipitationPreviousHourInches_mean_u', 'DiffuseHorizontalRadiation_mean_u', 'MslPressure_mean_u', 'SnowfallInches_mean_u', 'SurfaceWindGustsMph_mean_u', 'PDSI_u', 'GDD_u', 'Close_pcch_cc', 'Close_pcch_co', 'Price_pcch_wco']]
dfe.to_csv('M:\Thesis\Final Data\data_final_u.csv', index = False)


'''Find list of states and counties'''
klm = (dff.drop_duplicates(subset = ['stcofips'], keep = 'first'))[['State', 'County', 'stcofips']] 
klm = klm.sort_values(['State', 'County', 'stcofips'], ascending = [1, 1, 1])
klm.to_csv('M:\Thesis\Final Data\county_total.csv', index = False)

'''Plot Futures Price'''
df_price['Month'] = ((df_price['Date'].str.partition('-')[2]).str.partition('-')[0]).map(lambda x: x.strip()).astype(int)
df_price = df_price[(df_price['Month'] != 1) & (df_price['Month'] != 2) & (df_price['Month'] != 3) & (df_price['Month'] != 11) & (df_price['Month'] != 12)]
df_price.count()
df_price.to_csv('M:\Thesis\Results\\futures_price.csv', index = False)

plt.plot(df_price['Date'], df_price['Close_pcch'], label = "Price Change (%)") 
plt.xlabel('Date') 
plt.ylabel('Change in Closing Price (%)') 
plt.title('Figure: Historical Closing Price Change (%)') 
plt.legend() 
plt.show()


