# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:16:08 2019

@author: ssingh17
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, LSTM

'''Modeling'''
data_final = pd.read_csv('M:\Documents\Thesis\Final Data\data_final.csv')
data_final = data_final[(data_final['Year'] > 1980)]
data_final.head(10)
data_final.count()

'''Finding Correlation Coefficient among Weather variables and Soil variables'''
abc = data_final[['SurfaceTemperatureFahrenheit_mean', 'SurfaceDewpointTemperatureFahrenheit_mean', 'SurfaceWetBulbTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindChillTemperatureFahrenheit_mean', 'ApparentTemperatureFahrenheit_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DownwardSolarRadiation_mean', 'DiffuseHorizontalRadiation_mean', 'DirectNormalIrradiance_mean', 'MslPressure_mean', 'HeatIndexFahrenheit_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'SurfaceTemperatureFahrenheit_max', 'SurfaceTemperatureFahrenheit_min']]
fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(abc.corr(method='pearson'), annot=True, fmt='.1f', 
            cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig('M:\Documents\Thesis\weather_corr_result.png', bbox_inches='tight', pad_inches=0.0)

xyz = data_final[['ffd', 'sandtotal', 'silttotal', 'claytotal', 'om', 'bulkDensity', 'lep', 'caco3', 'ec', 'soc0_150', 'rootznaws', 'droughty', 'sand', 'share_cropland']]
fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(xyz.corr(method='pearson'), annot=True, fmt='.1f', 
            cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig('M:\Documents\Thesis\soil_corr_result.png', bbox_inches='tight', pad_inches=0.0)

'''Modeling part (Pre-processing)'''
data_final =data_final[['Year', 'Date_seq', 'SurfaceTemperatureFahrenheit_mean', 'RelativeHumidity_mean', 'SurfaceAirPressureMillibars_mean', 'CloudCoverage_mean', 'WindSpeedMph_mean', 'WindDirection_mean', 'PrecipitationPreviousHourInches_mean', 'DiffuseHorizontalRadiation_mean', 'MslPressure_mean', 'SnowfallInches_mean', 'SurfaceWindGustsMph_mean', 'PDSI', 'GDD', 'ffd', 'silttotal', 'om', 'bulkDensity', 'lep', 'caco3', 'ec', 'soc0_150', 'droughty', 'sand', 'share_cropland', 'Close_pcch']]
data_final_train = data_final[(data_final['Year'] < 2018)]
data_final_test = data_final[(data_final['Year'] >= 2018)]

data_train = data_final_train.iloc[:, 1:].values
data_test = data_final_test.iloc[:, 1:].values
sc = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = sc.fit_transform(data_train)
data_test_scaled = sc.fit_transform(data_test)

'''Finding numner of steps dynamically & Converting data into X(3-D) and y(2-D)'''
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

X_train, y_train = split_sequences(data_train_scaled)
X_test, y_test = split_sequences(data_test_scaled)  
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

'''Fully Connected Neural Network'''

X_train1, X_test1 = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2]), X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
model = Sequential()
model.add(Dense(512, activation = 'linear', input_dim = X_train1.shape[1]))
model.add(Dense(64, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.summary()
#model.fit(X_train1, y_train.ravel(), epochs=100, validation_data=(X_test1, y_test.ravel()), verbose=0, shuffle=False)
history = model.fit(X_train1, y_train, epochs=500, validation_data=(X_test1, y_test), verbose=0, shuffle=False)
model.evaluate(X_test1, y_test, verbose=2)

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
    
yhat = model.predict(X_test1)
n_steps = X_test.shape[1]
yhat_new = np.repeat(yhat, n_steps)
yhat_new = yhat_new.reshape(len(yhat_new), 1)
data_pred_scaled = np.concatenate((data_test_scaled[:, :-1], yhat_new), axis = 1)       
data_pred = sc.inverse_transform(data_pred_scaled) 

def series_y(data_df):
    y_new = list()
    i = 0
    while i < len(data_df):
        set_y = data_df[i, -1:]
        y_new.append(set_y)
        i = i + n_steps
    return np.array(y_new)
    
y_test_new = series_y(data_test)
y_pred_new = series_y(data_pred)

df_f_pred = pd.DataFrame({'Close_pcch_actual':y_test_new[:,0],'Close_pcch_predicted':y_pred_new[:,0]})
mse = mean_squared_error(df_f_pred.loc[:,'Close_pcch_actual'],df_f_pred.loc[:,'Close_pcch_predicted'])
print('MSE: %6f' % mse)
print(df_f_pred.head(20))
df_f_pred.to_csv('M:\Documents\Thesis\Results\df_f_pred.csv', index = False)

df_f_pred = df_f_pred.reset_index()
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_predicted'], label = "Predicted price change") 
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_actual'], label = "Actual price change")  
plt.xlabel('Index') 
plt.ylabel('Change in Closing Price (%)') 
plt.title('Figure: Actual vs Predicted percent change in Closing Price') 
plt.legend() 
plt.show()


'''Fully Connected Neural Network (incorporate temporal impact)'''

'To incorporate temporal effect, taking previous days using n_steps found as above & Converting data into X(3-D) and y(2-D)'
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    i = 0
    while i < len(sequences):
        i = i + n_steps
        end_ix = i + 4*n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i - n_steps:end_ix, 1:-1], sequences[end_ix - 1, -1:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 772
X_train, y_train = split_sequences(data_train_scaled, n_steps)
X_test, y_test = split_sequences(data_test_scaled, n_steps)  
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train1, X_test1 = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2]), X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
model = Sequential()
model.add(Dense(512, activation = 'linear', input_dim = X_train1.shape[1]))
model.add(Dense(64, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.summary()
#model.fit(X_train1, y_train.ravel(), epochs=100, validation_data=(X_test1, y_test.ravel()), verbose=0, shuffle=False)
history = model.fit(X_train1, y_train, epochs=500, validation_data=(X_test1, y_test), verbose=0, shuffle=False)
model.evaluate(X_test1, y_test, verbose=2)

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

n_steps_new = 5*n_steps
yhat = model.predict(X_test1)

'Extract date sequence column'
date_seq = list()
i = 0
while i < (len(data_test_scaled) - n_steps_new + 1):
    set_y = data_test_scaled[i + n_steps_new - 1, 0:1]
    date_seq.append(set_y)
    i = i + 772
date_seq = np.array(date_seq)

yhat_new = np.repeat(yhat, n_steps_new).reshape(len(yhat), n_steps_new)
date_seq_new = np.repeat(date_seq, n_steps_new).reshape(len(date_seq), n_steps_new)

abc = np.concatenate((date_seq_new[:, :, np.newaxis], X_test), axis=2)
data_pred_scaled = np.concatenate((abc, yhat_new[:, :, np.newaxis]), axis=2)

'Convert 3-D data set to 2-D'
z = list()
for i in range(len(data_pred_scaled)):
    for j in range(n_steps_new):
        z1 = data_pred_scaled[i][j]
        z.append(z1)
z = np.array(z)        
data_pred = sc.inverse_transform(z)    

'Find test y and predicted y to compare & calculate MSE' 
y_test_new = list()
i = 0
while i < (len(data_test) - n_steps_new + 1):
    set_y = data_test[i + n_steps_new - 1, -1:]
    y_test_new.append(set_y)
    i = i + 772
y_test_new = np.array(y_test_new)

y_pred_new = list()
i = 0
while i < len(data_pred):
    set_yp = data_pred[i, -1:]
    y_pred_new.append(set_yp)
    i = i + n_steps_new
y_pred_new = np.array(y_pred_new)
df_f_pred = pd.DataFrame({'Close_pcch_actual':y_test_new[:,0],'Close_pcch_predicted':y_pred_new[:,0]})
mse = mean_squared_error(df_f_pred.loc[:,'Close_pcch_actual'],df_f_pred.loc[:,'Close_pcch_predicted'])
print('MSE: %6f' % mse)
print(df_f_pred.tail(10))
df_f_pred.to_csv('M:\Documents\Thesis\Results\df_f_pred.csv', index = False)

df_f_pred = df_f_pred.reset_index()
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_predicted'], label = "Predicted price change") 
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_actual'], label = "Actual price change")  
plt.xlabel('Index') 
plt.ylabel('Change in Closing Price (%)') 
plt.title('Figure: Actual vs Predicted percent change in Closing Price') 
plt.legend() 
plt.show()


'''CNN-1D'''
n_steps, n_features = X_train.shape[1], X_train.shape[2] 
model = Sequential()
model.add(Conv1D(filters = 64, kernel_size = 3, activation = 'relu', input_shape = (n_steps, n_features)))
#model.add(Conv1D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (n_steps, n_features)))
model.add(BatchNormalization())
#model.add(Dropout(0.20))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'linear', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(512, activation = 'linear'))
model.add(Dense(64, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.summary()
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), verbose=0, shuffle=False)
model.evaluate(X_test, y_test, verbose=2)

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

yhat = model.predict(X_test)
n_steps = X_test.shape[1]
yhat_new = np.repeat(yhat, n_steps)
yhat_new = yhat_new.reshape(len(yhat_new), 1)
data_pred_scaled = np.concatenate((data_test_scaled[:, :-1], yhat_new), axis = 1)       
data_pred = sc.inverse_transform(data_pred_scaled)    

def series_y(data_df):
    y_new = list()
    i = 0
    while i < len(data_df):
        set_y = data_df[i, -1:]
        y_new.append(set_y)
        i = i + n_steps
    return np.array(y_new)
    
y_test_new = series_y(data_test)
y_pred_new = series_y(data_pred)

df_f_pred = pd.DataFrame({'Close_pcch_actual':y_test_new[:,0],'Close_pcch_predicted':y_pred_new[:,0]})
mse = mean_squared_error(df_f_pred.loc[:,'Close_pcch_actual'],df_f_pred.loc[:,'Close_pcch_predicted'])
print('MSE: %6f' % mse)
print(df_f_pred.tail(10))
df_f_pred.to_csv('M:\Documents\Thesis\Results\df_f_pred.csv', index = False)

df_f_pred = df_f_pred.reset_index()
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_predicted'], label = "Predicted price change") 
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_actual'], label = "Actual price change")  
plt.xlabel('Index') 
plt.ylabel('Change in Closing Price (%)') 
plt.title('Figure: Actual vs Predicted percent change in Closing Price') 
plt.legend() 
plt.show()


'''LSTM'''

'To incorporate temporal effect, taking previous days and present day (i.e. 10) using n_steps found as above & Converting data into X(3-D) and y(2-D) - past 9 days and present day data (X_t-9 to X_t) to predict present day price change (y_t)'
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    i = 0
    while i < len(sequences):
        end_ix = i + 10*n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i : end_ix, 1 : -1], sequences[end_ix - 1, -1:]
        X.append(seq_x)
        y.append(seq_y)
        i = i + n_steps
    return np.array(X), np.array(y)

n_steps = 772
X_train, y_train = split_sequences(data_train_scaled, n_steps)
X_test, y_test = split_sequences(data_test_scaled, n_steps)  
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

'Convert X(3-D) to X(4-D), stacking temporal data for every counties (i.e. temporal data sorted on time for county no. 1 then county no. 2 and likewise)'
def create_data(data, n_steps, time_steps):
    X = []
    for i in range(len(data)):
        B = []
        for j in range(n_steps):
            A = []
            for k in range(time_steps):
                a = data[i][j + n_steps*k]
                A.append(a)
            A = np.array(A)
            B.append(A)
        B = np.array(B)
        X.append(B)
    return np.array(X)
        
n_steps = 772
time_steps = 10
X_train_new = create_data(X_train, n_steps, time_steps)
X_test_new = create_data(X_test, n_steps, time_steps)
print(X_train_new.shape, y_train.shape, X_test_new.shape, y_test.shape)

time_steps, n_features = X_train_new.shape[2], X_train_new.shape[3] 
model = Sequential()
model.add(LSTM(64, return_sequences = False, input_shape = (time_steps, n_features)))
#model.add(Dropout(0.2))
#model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(Dense(1))
#opt = optimizers.RMSprop(lr=0.00001)
model.compile(optimizer='adam', loss='mse', metrics = ['mae'])
history = model.fit(X_train_new, y_train, epochs=1, batch_size=30, validation_data=(X_test_new, y_test), verbose=0, shuffle=False)
scores = model.evaluate(X_test_new, y_test, verbose=2)

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


'''CNN-1D + LSTM'''
'To incorporate temporal effect, taking previous days and present day (i.e. 10) using n_steps found as above & Converting data into X(3-D) and y(2-D) - past 9 days and present day data (X_t-9 to X_t) to predict present day price change (y_t)'
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    i = 0
    while i < len(sequences):
        end_ix = i + 10*n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i : end_ix, 1 : -1], sequences[end_ix - 1, -1:]
        X.append(seq_x)
        y.append(seq_y)
        i = i + n_steps
    return np.array(X), np.array(y)

n_steps = 772
X_train, y_train = split_sequences(data_train_scaled, n_steps)
X_test, y_test = split_sequences(data_test_scaled, n_steps)  
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

'Convert X(3-D) to X(4-D), stacking temporal data for every counties (i.e. temporal data sorted on time for county no. 1 then county no. 2 and likewise)'
def create_data(data, time_steps, n_steps):
    X = []
    for i in range(len(data)):
        B = []
        for j in range(time_steps):
            A = []
            for k in range(n_steps):
                a = data[i][n_steps*j + k]
                A.append(a)
            A = np.array(A)
            B.append(A)
        B = np.array(B)
        X.append(B)
    return np.array(X)
        
time_steps = 10
n_steps = 772
X_train_new = create_data(X_train, time_steps, n_steps)
X_test_new = create_data(X_test, time_steps, n_steps)
print(X_train_new.shape, y_train.shape, X_test_new.shape, y_test.shape)

time_steps, n_steps, n_features = X_train_new.shape[1], X_train_new.shape[2], X_train_new.shape[3] 
input_shape = (time_steps, n_steps, n_features)
model = Sequential()
model.add(TimeDistributed(Conv1D(filters = 64, kernel_size = 3, activation = 'linear', padding = 'same', input_shape = input_shape)))
model.add(TimeDistributed(BatchNormalization()))
#model.add(Dropout(0.20))
model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
model.add(TimeDistributed(Flatten()))
model.add(Dense(64, activation = 'linear'))
model.add(LSTM(64, return_sequences = True))
model.add(LSTM(10, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.build((None,) + input_shape)
model.summary()
history = model.fit(X_train_new, y_train, epochs=100, validation_data=(X_test_new, y_test), verbose=0, shuffle=False)
model.evaluate(X_test_new, y_test, verbose=2)

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

yhat = model.predict(X_test_new)

'Extract date sequence column'
n_steps_new = 10*n_steps
date_seq = list()
i = 0
while i < (len(data_test_scaled) - n_steps_new + 1):
    set_y = data_test_scaled[i + n_steps_new - 1, 0 : 1]
    date_seq.append(set_y)
    i = i + n_steps
date_seq = np.array(date_seq)

yhat_new = np.repeat(yhat, n_steps_new).reshape(len(yhat), n_steps_new)
date_seq_new = np.repeat(date_seq, n_steps_new).reshape(len(date_seq), n_steps_new)

abc = np.concatenate((date_seq_new[:, :, np.newaxis], X_test), axis=2)
data_pred_scaled = np.concatenate((abc, yhat_new[:, :, np.newaxis]), axis=2)

'Convert 3-D data set to 2-D'
z = list()
for i in range(len(data_pred_scaled)):
    for j in range(n_steps_new):
        z1 = data_pred_scaled[i][j]
        z.append(z1)
z = np.array(z)        
data_pred = sc.inverse_transform(z)  

'Find test y and predicted y to compare & calculate MSE' 
y_test_new = list()
i = 0
while i < (len(data_test) - n_steps_new + 1):
    set_y = data_test[i + n_steps_new - 1, -1:]
    y_test_new.append(set_y)
    i = i + n_steps
y_test_new = np.array(y_test_new)

y_pred_new = list()
i = 0
while i < len(data_pred):
    set_yp = data_pred[i, -1:]
    y_pred_new.append(set_yp)
    i = i + n_steps_new
y_pred_new = np.array(y_pred_new)
df_f_pred = pd.DataFrame({'Close_pcch_actual':y_test_new[:,0],'Close_pcch_predicted':y_pred_new[:,0]})
mse = mean_squared_error(df_f_pred.loc[:,'Close_pcch_actual'],df_f_pred.loc[:,'Close_pcch_predicted'])
print('MSE: %6f' % mse)
print(df_f_pred.tail(10))
df_f_pred.to_csv('M:\Documents\Thesis\Results\df_f_pred.csv', index = False)

df_f_pred = df_f_pred.reset_index()
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_predicted'], label = "Predicted price change") 
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_actual'], label = "Actual price change")  
plt.xlabel('Index') 
plt.ylabel('Change in Closing Price (%)') 
plt.title('Figure: Actual vs Predicted percent change in Closing Price') 
plt.legend() 
plt.show()






















'''CNN-1D + LSTM'''
X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]
n_steps, n_features, n_channels = X_train.shape[1], X_train.shape[2], X_train.shape[3] 
input_shape = (n_steps, n_features, n_channels)
model = Sequential()
model.add(TimeDistributed(Conv1D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = input_shape)))
model.add(TimeDistributed(BatchNormalization()))
#model.add(Dropout(0.20))
model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
model.add(TimeDistributed(Flatten()))
#model.add(LSTM(64, return_sequences = True))
model.add(LSTM(1))
model.add(Dense(512, activation = 'linear'))
model.add(Dense(64, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.build((None,) + input_shape)
model.summary()
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), verbose=0, shuffle=False)
model.evaluate(X_test, y_test, verbose=2)

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

yhat = model.predict(X_test)
n_steps = X_test.shape[1]
yhat_new = np.repeat(yhat, n_steps)
yhat_new = yhat_new.reshape(len(yhat_new), 1)
data_pred_scaled = np.concatenate((data_test_scaled[:, :-1], yhat_new), axis = 1)       
data_pred = sc.inverse_transform(data_pred_scaled)    

def series_y(data_df):
    y_new = list()
    i = 0
    while i < len(data_df):
        set_y = data_df[i, -1:]
        y_new.append(set_y)
        i = i + n_steps
    return np.array(y_new)
    
y_test_new = series_y(data_test)
y_pred_new = series_y(data_pred)

df_f_pred = pd.DataFrame({'Close_pcch_actual':y_test_new[:,0],'Close_pcch_predicted':y_pred_new[:,0]})
mse = mean_squared_error(df_f_pred.loc[:,'Close_pcch_actual'],df_f_pred.loc[:,'Close_pcch_predicted'])
print('MSE: %6f' % mse)
print(df_f_pred.tail(10))
df_f_pred.to_csv('M:\Documents\Thesis\Results\df_f_pred.csv', index = False)

df_f_pred = df_f_pred.reset_index()
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_predicted'], label = "Predicted price change") 
plt.plot(df_f_pred['index'], df_f_pred['Close_pcch_actual'], label = "Actual price change")  
plt.xlabel('Index') 
plt.ylabel('Change in Closing Price (%)') 
plt.title('Figure: Actual vs Predicted percent change in Closing Price') 
plt.legend() 
plt.show()


'''Convolution 2D'''
X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]
n_steps, n_features, n_channels = X_train.shape[1], X_train.shape[2], X_train.shape[3] 
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (n_steps, n_features, n_channels)))
#model.add(Conv1D(filters = 64, kernel_size = 3, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0, shuffle=False)
model.evaluate(X_test, y_test, verbose=2)

yhat = model.predict(X_test)
n_steps = X_test.shape[1]
yhat_new = np.repeat(yhat, n_steps)
yhat_new = yhat_new.reshape(len(yhat_new), 1)
data_pred_scaled = np.concatenate((data_test_scaled[:, :-1], yhat_new), axis = 1)       
data_pred = sc.inverse_transform(data_pred_scaled)    

def series_y(data_df):
    y_new = list()
    i = 0
    while i < len(data_df):
        set_y = data_df[i, -1:]
        y_new.append(set_y)
        i = i + n_steps
    return np.array(y_new)
    
y_test_new = series_y(data_test)
y_pred_new = series_y(data_pred)

df_f_pred = pd.DataFrame({'Close_pcch_actual':y_test_new[:,0],'Close_pcch_predicted':y_pred_new[:,0]})
mse = mean_squared_error(df_f_pred.loc[:,'Close_pcch_actual'],df_f_pred.loc[:,'Close_pcch_predicted'])
print('MSE: %6f' % mse)
print(df_f_pred.head(10))



























