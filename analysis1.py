# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:16:21 2019

@author: ssingh17
"""
import pandas as pd
import os
import sys 
import csv

'''Counting and Printing number of weather files in each counties of every states to find out when the weather data starts coming for that county'''
stdoutOrigin=sys.stdout 
sys.stdout = open(r'M:\Documents\Thesis\weatherfile_details.txt', 'w+')
path = 'M:/Documents/Thesis/Weather Data'
folders = []
for name in os.listdir(path):
    if os.path.isdir(os.path.join(path, name)) and not name.startswith('y'):
        folders.append(name)

for folder in folders:
    subfolders = []
    path = 'M:/Documents/Thesis/Weather Data' + '/' + folder
    print(path)
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            subfolders.append(name)
    print('No. of Counties: % 4d' % len(subfolders))
    for subfolder in subfolders:
        contents = os.listdir(os.path.join(path, subfolder))
        print(subfolder, len(contents))
sys.stdout.close()
sys.stdout=stdoutOrigin 

'''Finding which counties have stopped corn production and in which year, find a csv file of counties to be dropped'''
stdoutOrigin=sys.stdout 
sys.stdout = open(r'M:\Documents\Thesis\county_yield_details.txt', 'w+')
df_county_yield = data_final = pd.read_csv('M:\Documents\Thesis\county_yield.csv')
df_county_yield['State'] = df_county_yield['State'].str.title()
df_county_yield['County'] = df_county_yield['County'].str.title()
df_county_yield.loc[(df_county_yield['State'] == 'Missouri') & (df_county_yield['County'] == 'Ste Genevieve'), 'County'] = 'Ste. Genevieve'
df_county_yield = df_county_yield.sort_values(['State', 'County', 'Year'], ascending = [1, 1, 1])
years = []
j = 1980
while j < 2019:
    years.append(j)
    j = j + 1
a = df_county_yield['State'].unique()

with open('M:\Documents\Thesis\Final Data\county_keep.csv', 'w+') as file:
    fieldnames = ['State', 'County']
    writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    for x in a:
        df_x = df_county_yield.loc[df_county_yield['State'] == x]
        b = (df_county_yield.loc[df_county_yield['State'] == x])['County'].unique()
        print('....', 'State', x, '....', sep = ': ')
        i = 0
        abc = []
        county_list = []
        for y in b:
            df_x_y = df_x.loc[df_x['County'] == y]
            print(y, len(df_x_y), sep = ': ')
            if len(df_x_y) == len(years):
                i = i + 1
            k = []
            for year in years:
                if year not in df_x_y['Year'].values:
                    k.append(year)
            if len(k) != 0:
                print('Year',  k, sep = ': ')
            if len(k) < 1 and y!= 'Other (Combined) Counties':
                county_list.append(y)
            c = 0
            for k1 in k:
                if k1 in set(years).difference((df_x.loc[df_x['County'] == 'Other (Combined) Counties'])['Year'].unique()):
                    c = c + 1
            if c > 0 and y!= 'Other (Combined) Counties':
                info = y + '-' + str(c)
                abc.append(info)
        print('Total No. of Counties (including Others) in', x, len(b), sep = ': ')
        print('No. of Counties having production for all year in', x, i, sep = ': ')
        print('Counties with no records of itself and Other in an year', x, abc, sep = ': ')
        print('Counties with records for more than 34 years', x, county_list, sep = ': ')
        print('................................')
        county_list_st = []
        for m in range(len(county_list)):
            county_list_st.append(x)
        for i, j in zip(county_list_st, county_list):
            writer.writerow({'State': i, 'County': j})
sys.stdout.close()
sys.stdout=stdoutOrigin 

'''Finding counties, having corn yield higher than a benchmark level (yield >=200)'''
county_high_yield = pd.read_csv('M:\Documents\Thesis\county_yield.csv') 
county_high_yield['State'] = county_high_yield['State'].str.title()
county_high_yield['County'] = county_high_yield['County'].str.title()
county_high_yield = county_high_yield[(county_high_yield['Year'] >= 2010)]
county_high_yield = county_high_yield[(county_high_yield['County'] != 'Other (Combined) Counties')]
county_high_yield.loc[(county_high_yield['State'] == 'Missouri') & (county_high_yield['County'] == 'Ste Genevieve'), 'County'] = 'Ste. Genevieve'
county_high_yield = county_high_yield[['State', 'County', 'Value']]
county_high_yield = county_high_yield.groupby(['State', 'County'], as_index = False).mean()
county_high_yield = county_high_yield[(county_high_yield['Value'] >= 170)]
county_high_yield =county_high_yield[['State', 'County']]
county_high_yield = county_high_yield.sort_values(['State', 'County'], ascending = [1, 1])
county_high_yield.to_csv('M:\Documents\Thesis\Final Data\county_high_yield.csv', index = False)


#county_high_yield.count()
#county_high_yield['State'].unique()
























