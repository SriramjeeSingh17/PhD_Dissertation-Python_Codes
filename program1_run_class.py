# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:54:31 2020

@author: ssingh17
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
#from lazypredict.Supervised import lazyClassifier
from keras.models import Sequential
import keras
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, LSTM

'''Modeling - daily price change'''
data_final = pd.read_csv('M:\Documents\Thesis\Final Data\\data_final.csv')
data_final = data_final[(data_final['Year'] > 1980)]
#data_final = data_final[(data_final['State'] == 'Iowa') | (data_final['State'] == 'Illinois')]
data_final.count()
data_final.tail()
#data_final[2735:2740]

'''Modeling part (Pre-processing)'''

'daily price change'
data_final['class'] = np.where(data_final['Close_pcch_cc'] >= 0, 1, 0)
#data = data_final.iloc[:, 1:].values

print(data_final['class'].value_counts(normalize = True)*100)

'daily price change'
#data_final =data_final[['Year', 'Date_seq', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'GDD', 'class']] #correl < |0.3|
data_final =data_final[['Year', 'Date_seq', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DiffuseHorizontalRadiation_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'PDSI', 'GDD', 'class']] #correl< |0.7|
#data_final =data_final[['Year', 'Date_seq', 'SurfaceTemperatureFahrenheit_mean_u', 'RelativeHumidity_mean_u', 'WindSpeedMph_mean_u', 'WindDirection_mean_u', 'MslPressure_mean_u', 'SnowfallInches_mean_u', 'class']] #correl < |0.3|

'For weekly price change'
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DiffuseHorizontalRadiation_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'PDSI', 'GDD', 'class']] #correl< |0.7|
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'WindSpeedMph_mean', 'class']] #correl < |0.3|
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'GDD', 'class']] #correl < |0.3|
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_u', 'RelativeHumidity_mean_u', 'WindSpeedMph_mean_u', 'WindDirection_mean_u', 'MslPressure_mean_u', 'SnowfallInches_mean_u', 'class']] #correl < |0.3|

data = data_final.iloc[:, 1:].values

#data_final_train = data_final[(data_final['Year'] < 2017)]
#data_final_test = data_final[(data_final['Year'] >= 2017)]
#
#data_train = data_final_train.iloc[:, 1:].values
#data_test = data_final_test.iloc[:, 1:].values

'''Finding numner of steps dynamically & Converting data into X(3-D) and y(2-D)'''
'''
def split_sequences(sequences):
    X, y = list(), list()
    i = 0 
    k = 0        
    while i < (len(sequences)):
        n_steps = 0 
        while k < len(sequences):
            if (k + 1) < len(sequences) and sequences[k, 0] == sequences[k+1, 0]:
                n_steps = n_steps + 1
                k = k + 1
            else:
                break 
        n_steps = n_steps + 1
        k = k + 1
        i = i + n_steps     
        seq_x, seq_y = sequences[i - n_steps:i, 1:-1], sequences[i - 1, -1:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

#X_train, y_train = split_sequences(data_train)
#X_test, y_test = split_sequences(data_test)  
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X, y = split_sequences(data)
print(X.shape, y.shape)
'''

'''(incorporate temporal impact - use X_t-3 to X_t to predict y_t)'''

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    i = 0
    while i < len(sequences):
        end_ix = i + 4*n_steps #(X_t-4 to X_t+3: 3-(-4)+1 = 8, X_t-1 to X_t+3: 3-(-1)+1 = 5)
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i : end_ix, 1 : -1], sequences[i + 4*n_steps - 1, -1:] #(X_t-4: 5 (4+1), X_t-1: 2)
        X.append(seq_x)
        y.append(seq_y)
        i = i + n_steps
    return np.array(X), np.array(y)

n_steps = 456
#X_train, y_train = split_sequences(data_train, n_steps)
#X_test, y_test = split_sequences(data_test, n_steps)  
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X, y = split_sequences(data, n_steps)
print(X.shape, y.shape)

'The last sample ends at second last row, as there is no row after the last row to compare & gives error (weekly price change)'
'''
def split_sequences(sequences, n_steps, time_steps):
    X, y = list(), list()
    i = 0 
    while i < (len(sequences) -1):
        while i < len(sequences) - 1:
            i = i + 1
            if (i + 1) > (time_steps*n_steps) and sequences[i - 1, 0] != sequences[i, 0]:
                break
        seq_x, seq_y = sequences[i - time_steps*n_steps : i, 1:-1], sequences[i - 1, -1:]
        X.append(seq_x)
        y.append(seq_y)
        #print(i)
    return np.array(X), np.array(y)       

n_steps = 155
time_steps = 5
X, y = split_sequences(data, n_steps, time_steps)
print(X.shape, y.shape)
'''

'''Modeling - weekly weather data for weekly price change'''
data_final = pd.read_csv('M:\Documents\Thesis\Final Data\\data_final_week.csv')
data_final = data_final[(data_final['Year'] > 1980)]
data_final.count()

'''Finding Correlation Coefficient among weekly Weather variables'''
'''
abc = data_final[['SurfaceTemperatureFahrenheit_mean_wkmean', 'SurfaceDewpointTemperatureFahrenheit_mean_wkmean', 'SurfaceWetBulbTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'CloudCoverage_mean_wkmean', 'WindChillTemperatureFahrenheit_mean_wkmean', 'ApparentTemperatureFahrenheit_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'DownwardSolarRadiation_mean_wkmean', 'DiffuseHorizontalRadiation_mean_wkmean', 'DirectNormalIrradiance_mean_wkmean', 'MslPressure_mean_wkmean', 'HeatIndexFahrenheit_mean_wkmean', 'SnowfallInches_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'SurfaceTemperatureFahrenheit_max_wkmax', 'SurfaceDewpointTemperatureFahrenheit_max_wkmax', 'SurfaceWetBulbTemperatureFahrenheit_max_wkmax', 'RelativeHumidity_max_wkmax', 'SurfaceAirPressureMillibars_max_wkmax', 'CloudCoverage_max_wkmax', 'WindChillTemperatureFahrenheit_max_wkmax', 'ApparentTemperatureFahrenheit_max_wkmax', 'WindSpeedMph_max_wkmax', 'WindDirection_max_wkmax', 'PrecipitationPreviousHourInches_max_wkmax', 'DownwardSolarRadiation_max_wkmax', 'DiffuseHorizontalRadiation_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'MslPressure_max_wkmax', 'HeatIndexFahrenheit_max_wkmax', 'SnowfallInches_max_wkmax', 'SurfaceWindGustsMph_max_wkmax']]

fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(abc.corr(method='pearson'), annot=True, fmt='.1f', 
            cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig('M:\Documents\Thesis\weather_weekly_corr_result.png', bbox_inches='tight', pad_inches=0.0)
'''
'''Modeling part (Pre-processing)'''
#data_final = data_final[(data_final['Price_pcch_wco'] != 0)]
data_final['class'] = np.where(data_final['Price_pcch_wco'] >= 0, 1, 0)
print(data_final['class'].value_counts(normalize = True)*100)

data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'MslPressure_mean_wkmean', 'CloudCoverage_mean_wkmean', 'SurfaceTemperatureFahrenheit_max_wkmax', 'PrecipitationPreviousHourInches_mean_wkmean', 'RelativeHumidity_mean_wkmax', 'SnowfallInches_mean_wkmean', 'WindDirection_mean_wkmean', 'class']]
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'MslPressure_mean_wkmean', 'CloudCoverage_mean_wkmean', 'SurfaceTemperatureFahrenheit_max_wkmax', 'PrecipitationPreviousHourInches_mean_wkmean', 'RelativeHumidity_max_wkmax', 'SnowfallInches_mean_wkmean', 'WindDirection_mean_wkmean', 'DownwardSolarRadiation_mean_wkmean', 'class']]

#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'CloudCoverage_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'DownwardSolarRadiation_mean_wkmean', 'DiffuseHorizontalRadiation_mean_wkmean', 'MslPressure_mean_wkmean', 'SnowfallInches_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'RelativeHumidity_max_wkmax', 'CloudCoverage_max_wkmax', 'WindSpeedMph_max_wkmax', 'WindDirection_max_wkmax', 'PrecipitationPreviousHourInches_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'SnowfallInches_max_wkmax', 'SurfaceWindGustsMph_max_wkmax', 'class']] #corr > |0.8|
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'CloudCoverage_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'DiffuseHorizontalRadiation_mean_wkmean', 'MslPressure_mean_wkmean', 'SnowfallInches_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'RelativeHumidity_max_wkmax', 'CloudCoverage_max_wkmax', 'WindDirection_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'SurfaceWindGustsMph_max_wkmax', 'class']] #corr > |0.7|
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'RelativeHumidity_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'WindSpeedMph_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'MslPressure_mean_wkmean', 'SnowfallInches_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'RelativeHumidity_max_wkmax', 'CloudCoverage_max_wkmax', 'WindDirection_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'class']] #corr > |0.6|
#data_final =data_final[['Year', 'Date_week_seq', 'SurfaceTemperatureFahrenheit_mean_wkmean', 'SurfaceAirPressureMillibars_mean_wkmean', 'WindDirection_mean_wkmean', 'PrecipitationPreviousHourInches_mean_wkmean', 'MslPressure_mean_wkmean', 'SurfaceWindGustsMph_mean_wkmean', 'RelativeHumidity_max_wkmax', 'CloudCoverage_max_wkmax', 'WindDirection_max_wkmax', 'DirectNormalIrradiance_max_wkmax', 'class']] #exclude var when corr > |0.4|

data = data_final.iloc[:, 1:].values

#data_final_train = data_final[(data_final['Year'] < 2012)]
#data_final_test = data_final[(data_final['Year'] >= 2012)]
#
#data_train = data_final_train.iloc[:, 1:].values
#data_test = data_final_test.iloc[:, 1:].values

'''Finding numner of steps dynamically & Converting data into X(3-D) and y(2-D), using X_t to predict y_t'''
'''
def split_sequences(sequences):
    X, y = list(), list()
    i = 0 
    k = 0        
    while i < (len(sequences)):
        n_steps = 0 
        while k < len(sequences):
            if (k + 1) < len(sequences) and sequences[k, 0] == sequences[k+1, 0]:
                n_steps = n_steps + 1
                k = k + 1
            else:
                break 
        n_steps = n_steps + 1
        k = k + 1
        i = i + n_steps     
        seq_x, seq_y = sequences[i - n_steps:i, 1:-1], sequences[i - 1, -1:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

#X_train, y_train = split_sequences(data_train)
#X_test, y_test = split_sequences(data_test)  
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X, y = split_sequences(data)
print(X.shape, y.shape)
'''

'''Finding numner of steps dynamically & Converting data into X(3-D) and y(2-D), using X_t-1 to predict y_t'''
'''
def split_sequences(sequences):
    X, y = list(), list()
    i = 0 
    k = 0        
    while i < (len(sequences)):
        n_steps = 0 
        while k < len(sequences):
            if (k + 1) < len(sequences) and sequences[k, 0] == sequences[k+1, 0]:
                n_steps = n_steps + 1
                k = k + 1
            else:
                break 
        n_steps = n_steps + 1
        k = k + 1
        i = i + n_steps     
        if i == len(sequences):
            break
        seq_x, seq_y = sequences[i - n_steps:i, 1:-1], sequences[i, -1:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = split_sequences(data)
print(X.shape, y.shape) 
'''

'''Converting data into X(3D) and y(2D) - using X_t+1 to predict y_t'''

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    i = 0
    while i < len(sequences) - n_steps:
        end_ix = i + n_steps 
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[end_ix : end_ix + n_steps, 1 : -1], sequences[i + n_steps - 1, -1:] 
        X.append(seq_x)
        y.append(seq_y)
        i = i + n_steps
    return np.array(X), np.array(y)

n_steps = 456
#X_train, y_train = split_sequences(data_train, n_steps)
#X_test, y_test = split_sequences(data_test, n_steps)  
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X, y = split_sequences(data, n_steps)
print(X.shape, y.shape)


'''Model Training'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train1, X_test1 = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2]), X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
print(X_train1.shape, y_train.shape, X_test1.shape, y_test.shape)


'Principal component analysis (PCA)'
#pca = PCA().fit(X_train2)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');

#pca = PCA(n_components = None)
pca = PCA(n_components = 1000)
X_train1 = pca.fit_transform(X_train1)
X_test1 = pca.transform(X_test1)
print(X_train1.shape, y_train.shape, X_test1.shape, y_test.shape)


'K-means Clustering'
model = KMeans(n_clusters=3, random_state = 42)
train_labels = model.fit_predict(X_train1)
X_train_c = np.concatenate((X_train1, train_labels[:, None]), axis=1)

test_labels = model.predict(X_test1)
X_test_c = np.concatenate((X_test1, test_labels[:, None]), axis=1)

sc = MinMaxScaler(feature_range=(0, 1))
X_train2 = sc.fit_transform(X_train_c)
X_test2 = sc.transform(X_test_c)

train = np.concatenate((X_train2, y_train), axis=1)
test = np.concatenate((X_test2, y_test), axis=1)

X_train2_0 = train[:, :-2][train[:, -2] == 0]
X_test2_0 = test[:, :-2][test[:, -2] == 0]
X_train2_1 = train[:, :-2][train[:, -2] == 0.5]
X_test2_1 = test[:, :-2][test[:, -2] == 0.5]
X_train2_2 = train[:, :-2][train[:, -2] == 1]
X_test2_2 = test[:, :-2][test[:, -2] == 1]
#print(X_train2_0.shape, X_test2_0.shape, X_train2_1.shape, X_test2_1.shape)
print(X_train2_0.shape, X_test2_0.shape, X_train2_1.shape, X_test2_1.shape, X_train2_2.shape, X_test2_2.shape)
y_train_0 = train[:, -1][train[:, -2] == 0]
y_test_0 = test[:, -1][test[:, -2] == 0]
y_train_1 = train[:, -1][train[:, -2] == 0.5]
y_test_1 = test[:, -1][test[:, -2] == 0.5]
y_train_2 = train[:, -1][train[:, -2] == 1]
y_test_2 = test[:, -1][test[:, -2] == 1]
#print(y_train_0.shape, y_test_0.shape, y_train_1.shape, y_test_1.shape)
print(y_train_0.shape, y_test_0.shape, y_train_1.shape, y_test_1.shape, y_train_2.shape, y_test_2.shape)


test.shape
test[-10:, -3:]

X_test2_0.shape
X_test2_0[-10:, -3:]


'Data Scaling'
sc = MinMaxScaler(feature_range=(0, 1))
X_train2 = sc.fit_transform(X_train1)
X_test2 = sc.transform(X_test1)

'''Logistic Regression - using clustered data'''
model = LogisticRegression(solver = 'liblinear') 
model_fit = model.fit(X_train2_1, y_train_1) 
y_pred = model_fit.predict(X_test2_1) 

print('y_train', Counter(y_train_1), 'y_test', Counter(y_test_1), 'y_pred', Counter((y_pred.round())), sep = ': ')
print(confusion_matrix(y_test_1, y_pred))  
print(classification_report(y_test_1, y_pred))
print("Accuracy:", accuracy_score(y_test_1, y_pred))
print("Precision:", precision_score(y_test_1, y_pred))

'''Logistic Regression'''
model = LogisticRegression(solver = 'liblinear') 
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

'''Support Vector Machine - using clustered data'''
model = SVC(kernel = 'sigmoid', gamma = 'auto') 
model_fit = model.fit(X_train2_0, y_train_0) 
y_pred = model_fit.predict(X_test2_0) 

print('y_train', Counter(y_train_0), 'y_test', Counter(y_test_0), 'y_pred', Counter((y_pred.round())), sep = ': ')
print(confusion_matrix(y_test_0, y_pred))  
print(classification_report(y_test_0, y_pred))
print("Accuracy:", accuracy_score(y_test_0, y_pred))
print("Precision:", precision_score(y_test_0, y_pred))

'''Support Vector Machine'''
#model = SVC(kernel='linear', gamma = 'auto', coef0 = 0.01, degree = 5, random_state = 0) 
model = SVC(kernel = 'sigmoid', gamma = 'auto') 
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

'''Random Forest'''
model = RandomForestClassifier(n_estimators = 1000)
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

'''Naive Bayes'''
model = GaussianNB()
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

'''Boosting - AdaBoost'''
model = AdaBoostClassifier(n_estimators = 1000)
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

'''Boosting - GradientBoost'''
#model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model = GradientBoostingClassifier(n_estimators=1000)
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

'''Boosting - XGBoost - using clustered data'''
model = XGBClassifier(booster = 'gblinear', n_estimators=1000)
model_fit = model.fit(X_train2_2, y_train_2) 
y_pred = model_fit.predict(X_test2_2) 

print('y_train', Counter(y_train_2), 'y_test', Counter(y_test_2), 'y_pred', Counter((y_pred.round())), sep = ': ')
print(confusion_matrix(y_test_2, y_pred))  
print(classification_report(y_test_2, y_pred))
print("Accuracy:", accuracy_score(y_test_2, y_pred))
print("Precision:", precision_score(y_test_2, y_pred))

'''Boosting - XGBoost'''
model = XGBClassifier(booster = 'gblinear', n_estimators=5000)
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))




'''CNN 1D'''
n_steps_new, n_features = X_train.shape[1], X_train.shape[2]
model = Sequential()
model.add(Conv1D(filters = 512, kernel_size = 5, activation = 'linear', padding = 'same', input_shape = (n_steps_new, n_features)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(filters = 256, kernel_size = 3, activation = 'linear', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))
#model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'linear', padding = 'same'))
#model.add(BatchNormalization())
#model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(256, activation = 'linear'))
model.add(Dense(64, activation = 'linear'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

#keras.optimizers.RMSprop(lr = 0.001)
#opt = RMSprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.Adam()
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=0, shuffle=False)
model.evaluate(X_test, y_test, verbose=2)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

y_pred = model.predict(X_test)

print(confusion_matrix(y_test.ravel(), y_pred.round()))  
print(classification_report(y_test.ravel(), y_pred.round()))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred.round()))
print("Precision:", precision_score(y_test.ravel(), y_pred.round()))



'''
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2)

print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2)

print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'newton-cg')
model_fit = model.fit(X_train2, y_train.ravel()) 
y_pred = model_fit.predict(X_test2) 

print('y_train', Counter(y_train.ravel()), 'y_test', Counter(y_test.ravel()), 'y_pred', Counter((y_pred.round()).ravel()), sep = ': ')
print(confusion_matrix(y_test.ravel(), y_pred))  
print(classification_report(y_test.ravel(), y_pred))
print("Accuracy:", accuracy_score(y_test.ravel(), y_pred))
print("Precision:", precision_score(y_test.ravel(), y_pred))

#scores = cross_val_score(model, X_train2, y_train.ravel(), cv=5)
#scores
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





