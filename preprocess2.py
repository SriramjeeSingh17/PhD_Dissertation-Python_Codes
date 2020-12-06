# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:57:28 2019

@author: ssingh17
"""

import pandas as pd
import glob
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import time

'''Illinois Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Illinois/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Illinois/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Illinois_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))


'''Indiana Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Indiana/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Indiana/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Indiana_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''Kansas Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Kansas/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Kansas/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Kansas_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''Michigan Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Michigan/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Michigan/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Michigan_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''Minnesota Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Minnesota/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Minnesota/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Minnesota_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''Missouri Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Missouri/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Missouri/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Missouri_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''Nebraska Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Nebraska/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Nebraska/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Nebraska_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''Ohio Weather Data''' 
os.chdir('M:/Thesis/Weather Data/Ohio/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/Ohio/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\Ohio_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))

'''South_Dakota Weather Data''' 
os.chdir('M:/Thesis/Weather Data/South_Dakota/')
counties = []
for name in os.listdir("."):
    if os.path.isdir(name):
        counties.append(name)
        
start_time = time.time()
df1 = pd.DataFrame()
for x in counties:
    for f in glob.glob('M:/Thesis/Weather Data/South_Dakota/' + x +'/'+x+'-' + '*.csv'):
        df2 = pd.read_csv(f)
        df2['County'] = x
        df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\South_Dakota_weather.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))


'''Weather Data after 2016'''
path = 'M:/Thesis/Weather Data/weather/'
files = []
for name in os.listdir(path):
    if '2016' not in name and '2019' not in name:
    #if not name.startswith('IA') and not name.startswith('SD'):
        files.append(name)

start_time = time.time()
df1 = pd.DataFrame()
for x in files:
    df2 = pd.read_csv(path + x)
    df2['state']= x[:2]
    df1 = df1.append(df2, sort = False, ignore_index=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
df1.to_hdf('M:\Thesis\Final Data\weather_current.h5', key = 'df1')
print("--- %s seconds ---" % (time.time() - start_time))
















