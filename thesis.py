#!/usr/bin/env python
# coding: utf-8

# In[1]:
import networkx as nx
import numpy as np
#import os
import csv
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
#import random
#import collections
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

import math
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# In[1]:


# In[3]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    df = data
    n_vars = df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    print(df.columns)
    if 'pc' in df.columns:
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        
            for i in range(0, n_out):
                cols.append(df['pc'].shift(-i))
                if i == 0:
                    names.append('pc')
                else:
                    names.append('pc(t+{0})'.format(i))
            # put it all together
            agg = pd.concat(cols, axis=1)
        
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

    else:
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    
            for i in range(0, n_out):
                cols.append(df['veh_acceleration'].shift(-i))
                if i == 0:
                    names.append('veh_acceleration')
                else:
                    names.append('veh_acceleration(t+{0})'.format(i))
            # put it all together
            agg = pd.concat(cols, axis=1)
        
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg





# In[43]:


def read_file(filename):
    #filename = 'C:\\fakultet\\Gjorgji Goran Reinforecemt\\Data_preprocessed\\DataSample10\\DataSample-10_freq-1.csv'
    #C:\fakultet\Gjorgji Goran Reinforecemt\Vremenska_serija_power_consumption
    #filename_plot = filename.replace('.csv',',')
    if "N12.csv" in filename:
        data = pd.read_csv(filename, sep = ";") 
        #data = pd.read_csv(filename, sep=',')
        data.set_index(['stamp'], inplace = True)
        del data['_id']
        del data['stamp_db']
        del data['node_id']
        return data
    else:
        data = pd.read_csv(filename, sep = ",") 
        #data = pd.read_csv(filename, sep=',')
        data.set_index(['DateTime'], inplace = True)
        del data['veh_speed-1']
        del data['veh_speed-2']
        del data['veh_acceleration_last']
        return data





# In[92]:


def LSTM_test(file_location, path,n_lag, n_to_predict, type_data):
#    n_lag = 1
    data = read_file(file_location)
    n_features = len(data.columns)
#    n_to_predict = 24 #current timestamp
    #path = 'C:\\fakultet\\Gjorgji Goran Reinforecemt\\Data_preprocessed\\DataSample10\\freq1\\'
    #print(train.head())

    reframed = series_to_supervised(data,n_lag,n_to_predict) 
    train_siz = int(len(reframed)*.9)
    test_siz = len(reframed)- train_siz
    train, test = reframed[0:train_siz], reframed[train_siz:len(reframed)]
    scaler = MinMaxScaler()
    #agg = scaler.fit_transform(agg)
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    
    units = train_siz//(2*(n_features-1 + n_to_predict))
    
    n_lstm_features = n_lag * n_features

    scaled_train_X, scaled_train_y = scaled_train[:,:n_lstm_features], scaled_train[:,n_lstm_features:]
    scaled_test_X , scaled_test_y = scaled_test[:,:n_lstm_features], scaled_test[:,n_lstm_features:]

    scaled_train_X = scaled_train_X.reshape((scaled_train_X.shape[0], n_lag, n_features))
    #print('train_reshaped shapes: ', scaled_X.shape, scaled_y.shape)
    scaled_test_X = scaled_test_X.reshape((scaled_test_X.shape[0], n_lag, n_features)) 
    bs = 0
    if len(train)<2000:
        bs = 32
    elif len(train)<5000:
        bs = 64
    elif len(train)<10000:
        bs = 128
    else:
        bs = 256
    
    model_l = Sequential()
    # units: 1,4,24
    #kodot da mu go pratam na profesorot
    model_l.add(LSTM(units,activation='relu',input_shape=(scaled_train_X.shape[1], scaled_train_X.shape[2])))
    model_l.add(Dense(n_to_predict))
    early = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 5)
    checkpoint = ModelCheckpoint(path + type_data+'LSTM-model-{epoch:03d}-{loss:03f}-{val_loss:03f}.hdf5', 
                                 verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
    model_l.compile(optimizer='adam', loss='mse',  metrics=['mse'])
    history = model_l.fit(scaled_train_X, scaled_train_y, validation_split=0.2,
                          epochs=60, batch_size = bs, verbose = 1, callbacks=[checkpoint, early])

    model_l.summary()
    
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path+ type_data+"LSTM_loss_"+str(n_lag)+"_pred_"+ str(n_to_predict)+".png")
    
    scaled_test_yhat = model_l.predict(scaled_test_X)
    scaled_test_X = scaled_test_X.reshape((scaled_test_X.shape[0],
                                          scaled_test_X.shape[1]*scaled_test_X.shape[2]))

    scaled_test_pred = np.concatenate((scaled_test_X[:,:], scaled_test_yhat), axis = 1)
    unscaled_test_pred = scaler.inverse_transform(scaled_test_pred)

    #print(unscaled_test_pred)
    test_yhat = unscaled_test_pred[:,-n_to_predict]


    test_y = test.iloc[:,-n_to_predict]
    #print(test_yhat[0])
    #print(test_y[0])
    testScore =math.sqrt(mean_squared_error(test_yhat, test_y.values))
    a = str.format("Test_score_%.4f_RMSE_" % (testScore))
    model_l.save(path+ a+ type_data + " LSTM_"+str(n_lag)+"_pred_"+ str(n_to_predict)+".h5")

    plt.figure(figsize=(25, 10))
    plt.title(a)
    plt.plot(test_yhat[0:test_siz], 'r') # plotting t, a separately 
    plt.plot(test_y.values[0:test_siz], 'b') # plotting t, b separately
    plt.savefig(path+a + type_data+ " RELU_LSTM_"+str(n_lag)+"_pred_"+ str(n_to_predict)+".png")
    plt.show()
    print(a)
    return testScore


# In[91]:


def GRU_test( file_location, path, n_lag, n_to_predict, type_data):
#    n_lag = 1
    data = read_file(file_location)
    n_features = len(data.columns)
#    n_to_predict = 24 #current timestamp
    #path = 'C:\\fakultet\\Gjorgji Goran Reinforecemt\\Data_preprocessed\\DataSample10\\freq1\\'
    #print(train.head())

    reframed = series_to_supervised(data,n_lag,n_to_predict) 
    train_siz = int(len(reframed)*.9)
    test_siz = len(reframed)- train_siz
    train, test = reframed[0:train_siz], reframed[train_siz:len(reframed)]
    scaler = MinMaxScaler()
    #agg = scaler.fit_transform(agg)
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    units = train_siz//(2*(n_features-1 + n_to_predict))
    
    n_lstm_features = n_lag * n_features

    scaled_train_X, scaled_train_y = scaled_train[:,:n_lstm_features], scaled_train[:,n_lstm_features:]
    scaled_test_X , scaled_test_y = scaled_test[:,:n_lstm_features], scaled_test[:,n_lstm_features:]

    scaled_train_X = scaled_train_X.reshape((scaled_train_X.shape[0], n_lag, n_features))
    #print('train_reshaped shapes: ', scaled_X.shape, scaled_y.shape)
    scaled_test_X = scaled_test_X.reshape((scaled_test_X.shape[0], n_lag, n_features)) 
    
    if len(train)<2000:
        bs = 32
    elif len(train)<5000:
        bs = 64
    elif len(train)<10000:
        bs = 128
    else:
        bs = 256
    
    model_l = Sequential()
    model_l.add(GRU(units, activation='relu', input_shape=(scaled_train_X.shape[1], scaled_train_X.shape[2])))
    model_l.add(Dense(n_to_predict))
    early = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 5)
    checkpoint = ModelCheckpoint(path + type_data+'GRU-model-{epoch:03d}-{loss:03f}-{val_loss:03f}.hdf5', 
                                 verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
    model_l.compile(optimizer='adam', loss='mse',  metrics=['mse'])
    history = model_l.fit(scaled_train_X, scaled_train_y, validation_split=0.2,
                          epochs=60, batch_size = bs, verbose = 1, callbacks=[checkpoint, early])

    model_l.summary()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path+ type_data+"GRU_loss_"+str(n_lag)+"_pred_"+ str(n_to_predict)+".png")
    
    scaled_test_yhat = model_l.predict(scaled_test_X)
    scaled_test_X = scaled_test_X.reshape((scaled_test_X.shape[0],
                                          scaled_test_X.shape[1]*scaled_test_X.shape[2]))

    scaled_test_pred = np.concatenate((scaled_test_X[:,:], scaled_test_yhat), axis = 1)
    unscaled_test_pred = scaler.inverse_transform(scaled_test_pred)

    #print(unscaled_test_pred)
    test_yhat = unscaled_test_pred[:,-n_to_predict]


    test_y = test.iloc[:,-n_to_predict]
    #print(test_yhat[0])
    #print(test_y[0])
    testScore =math.sqrt(mean_squared_error(test_yhat, test_y.values))
    a = str.format("Test_score_%.4f_RMSE_" % (testScore))
    #model_l.save(path+ a+ type_data+ " GRU_"+str(n_lag)+"_pred_"+ str(n_to_predict)+".h5")

    plt.figure(figsize=(25, 10))
    plt.title(a)
    plt.plot(test_yhat[0:test_siz], 'r') # plotting t, a separately 
    plt.plot(test_y.values[0:test_siz], 'b') # plotting t, b separately
    plt.savefig(path+a + type_data+" RELU_GRU_"+str(n_lag)+"_pred_"+ str(n_to_predict)+".png")
    plt.show()
    print(a)
    return testScore
# In[358]:

from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

# In[358]:

def random_walk(data, n_to_predict):
    data = read_file(data)
    train_siz = int(len(data)*.9)
    test_siz = len(data)- train_siz
    train, test = data[0:train_siz], data[train_siz:len(data)]
    test = test['veh_acceleration']
    test.dropna(inplace=True)
    test_ = []

    for i in range(0,len(test), n_to_predict+1):
        test_[i:i+n_to_predict] = [test[i]]*(n_to_predict+1)
    l = []
    for i in range(len(test)):
        l.append(test_[i])

    test_siz = len(test)
    testScore =math.sqrt(mean_squared_error(test, l))
    a = str.format("Test_score_%.4f_RMSE_" % (testScore))
#    plt.figure(figsize=(25, 10))
#    plt.title(a)
#    plt.plot(test.values, 'r') # plotting t, a separately 
#    plt.plot(l, 'b') # plotting t, b separately
#    plt.show()
    print(len(test))
    print(a)


# ## Statistical approaches:

# In[307]:
def moving_average(test, n_to_predict):
    test.dropna(inplace=True)
    test_yhat = test.iloc[n_to_predict:]
    test_y = test.iloc[:-n_to_predict]
    test_siz = len(test)
    testScore =math.sqrt(mean_squared_error(test_yhat.values, test_y.values))
    a = str.format("Test_score_%.4f_RMSE_" % (testScore))
    #plt.figure(figsize=(25, 10))
    #plt.title(a)
    #plt.plot(test_yhat.values[0:test_siz], 'r') # plotting t, a separately 
    #plt.plot(test_y.values[0:test_siz], 'b') # plotting t, b separately
    #plt.show()
    print(a)

# In[30]:


def arima_results(file_location,path ,lag,n_forecast, type_data):
    
    data = read_file(file_location)
    data.dropna(inplace=True)
    train_siz = int(len(data)*.9)
    test_siz = len(data)- train_siz
    train, test = data[0:train_siz], data[train_siz:len(data)]
    
    print(lag,n_forecast)
    if "N12.csv" in file_location:
        history = train['pc']
        predictions = list()
        error = 0
        obs = None
        for t in range(len(test['pc'].values)-(n_forecast-1)):
            model = ARIMA(history, order=(3,0,2))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast(n_forecast)
            if t==0 and n_forecast != 1:
                yhat = output[0]
                predictions.extend(yhat)
            elif n_forecast == 1:
                yhat = output[0]
                predictions.extend(yhat)
            else:
                yhat = output[0][-1]
                predictions.append(yhat)
            
            if n_forecast == 1:
                obs = test['pc'].iloc[t]
            else:
                obs = test['pc'].iloc[t:t+n_forecast].values
            history = history.append(pd.Series(obs), ignore_index=True)
        error = math.sqrt(mean_squared_error(test['pc'].values, predictions))
        a = str.format("Test_score_%.4f_RMSE_" % (error))
        plt.figure(figsize=(25, 10))
        plt.title(a)
        plt.plot(test['pc'].values, color = 'blue')
        plt.plot(predictions, color='red')
        plt.savefig(path + a + type_data+ " ARMA_lag_"+str(lag)+"_pred_"+str(n_forecast)+".png")
        plt.show()
        return error

    else:    
        history = train['veh_acceleration']
        predictions = list()
        error = 0
        obs = None
        for t in range(len(test['veh_acceleration'].values)-(n_forecast-1)):
            model = ARIMA(history, order=(3,0,2))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast(n_forecast)
            if t==0 and n_forecast != 1:
                yhat = output[0]
                predictions.extend(yhat)
            elif n_forecast == 1:
                yhat = output[0]
                predictions.extend(yhat)
            else:
                yhat = output[0][-1]
                predictions.append(yhat)
            
            if n_forecast == 1:
                obs = test['veh_acceleration'].iloc[t]
            else:
                obs = test['veh_acceleration'].iloc[t:t+n_forecast].values
            history = history.append(pd.Series(obs), ignore_index=True)
        error = math.sqrt(mean_squared_error(test['veh_acceleration'].values, predictions))
        a = str.format("Test_score_%.4f_RMSE_" % (error))
        plt.figure(figsize=(25, 10))
        plt.title(a)
        plt.plot(test['veh_acceleration'].values, color = 'blue')
        plt.plot(predictions, color='red')
        plt.savefig(path + a + type_data+ " ARMA_lag_"+str(lag)+"_pred_"+str(n_forecast)+".png")
        plt.show()
        return error


# In[31]:


def ar_results(file_location,path,lag,n_forecast, type_data):
    data = read_file(file_location)
    print(lag,n_forecast)
    data.dropna(inplace=True)
    train_siz = int(len(data)*.9)
    test_siz = len(data)- train_siz
    train, test = data[0:train_siz], data[train_siz:len(data)]
    if "N12.csv" in file_location:
        history = [x for x in train['pc']]
        predictions = list()
        error = 0
        obs = None
        for t in range(len(test['pc'].values)-(n_forecast-1)):
            model = AR(history)
            model_fit = model.fit(maxlag=lag, disp=0)
            output = model_fit.predict(start = len(history), 
                                       end = len(history) + n_forecast-1)
            
            if t==0 and n_forecast != 1:
                yhat = output
                predictions.extend(yhat)
            elif n_forecast == 1:
                yhat = output
                predictions.extend(yhat)
            else:
                yhat = output[-1]
                predictions.append(yhat)
            
            if n_forecast == 1:
                obs = test['pc'].values[t]
                history.append(obs)
            else:
                obs = test['pc'].values[t:t+n_forecast]
                history.extend(obs)
    
        error = math.sqrt(mean_squared_error(test['pc'].values, predictions))
        a = str.format("Test_score_%.4f_RMSE_" % (error))
        plt.figure(figsize=(25, 10))
        plt.title(a)
        plt.plot(test['pc'].values, color = 'blue')
        plt.plot(predictions, color='red')
        plt.savefig(path + a + type_data + " AR_lag_"+str(lag)+"_pred_"+str(n_forecast)+".png")
        plt.show()
        return error

    else:
        history = [x for x in train['veh_acceleration']]
        predictions = list()
        error = 0
        obs = None
        for t in range(len(test['veh_acceleration'].values)-(n_forecast-1)):
            model = AR(history)
            model_fit = model.fit(maxlag=lag, disp=0)
            output = model_fit.predict(start = len(history), 
                                       end = len(history) + n_forecast-1)
            
            if t==0 and n_forecast != 1:
                yhat = output
                predictions.extend(yhat)
            elif n_forecast == 1:
                yhat = output
                predictions.extend(yhat)
            else:
                yhat = output[-1]
                predictions.append(yhat)
            
            if n_forecast == 1:
                obs = test['veh_acceleration'].values[t]
                history.append(obs)
            else:
                obs = test['veh_acceleration'].values[t:t+n_forecast]
                history.extend(obs)
    
        error = math.sqrt(mean_squared_error(test['veh_acceleration'].values, predictions))
        a = str.format("Test_score_%.4f_RMSE_" % (error))
        plt.figure(figsize=(25, 10))
        plt.title(a)
        plt.plot(test['veh_acceleration'].values, color = 'blue')
        plt.plot(predictions, color='red')
        plt.savefig(path + a + type_data + " AR_lag_"+str(lag)+"_pred_"+str(n_forecast)+".png")
        plt.show()
        return error        
    

# In[90]:


list_dir = []
list_file = []
file_name = "/home/petre/janeDiplomska/results.txt"
for l in range(0,4,1):
    a = str.format("/home/petre/janeDiplomska/DataSample-1%d_freq-0.1.csv" % (l))
    list_file.append(a)
    a = str.format("/home/petre/janeDiplomska/DataSample-1%d_freq-0.5.csv" % (l))
    list_file.append(a)
    a = str.format("/home/petre/janeDiplomska/DataSample-1%d_freq-1.csv" % (l))
    list_file.append(a)
    a = "/home/petre/janeDiplomska/"
    list_dir.append(a)
    a = "/home/petre/janeDiplomska/" 
    list_dir.append(a)
    a = "/home/petre/janeDiplomska/"
    list_dir.append(a)
#    a = "C:\\fakultet\\Gjorgji Goran Reinforecemt\\Vremenska_serija_power_consumption\\N12.csv"
#    list_file.append(a)    
#    a = "C:\\fakultet\\Gjorgji Goran Reinforecemt\\Vremenska_serija_power_consumption\\"
#    list_dir.append(a)
#(1,1),(1,4),(1,24),(3,1),(3,4),(3,24),(5,1),(5,4),(5,24),(7,1),(7,4),(7,24),(9,1),(9,4),(9,24)
li = [(1,1),(1,4),(1,24)]#,(3,1),(3,4),(3,24),(5,1),(5,4),(5,24),(7,1),(7,4),(7,24),(9,1),(9,4),(9,24)]
l_res = []
for lf,ld in zip(list_file, list_dir):
    for l in li:
#        l_res.append([str("LSTM " + str(l) + str(lf[-6:-4])),str(LSTM_test(lf,ld,l[0],l[1], lf[-6:-4])),"\n"])
#        l_res.append([str("GRU " + str(l) + str(lf[-6:-4])),str(GRU_test(lf,ld,l[0],l[1], lf[-6:-4])),"\n"])
        l_res.append([str("ARMA+ " + str(l)+ str(lf[-6:-4])),str(arima_results(lf,ld,l[0],l[1], lf[-6:-4])),"\n"])
        #print(l)
        #random_walk(lf, l[1])
#        l_res.append([str("AR " + str(l) + str(lf[-6:-4])),str(ar_results(lf,ld,l[0],l[1], lf[-6:-4])),"\n"])
with open(file_name,'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(l_res)
#print(l_res)      
#za (9,1) nemam statistickite a drugite nisto
