# -*- coding: utf-8 -*-
"""

"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot 
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU


#construct a time series which contains:
# stock price and sentiment score from the yearsteday
# stock price from today 



def sentiment_stockPrice_series(sentimentScore, stockPrice, scaler):
    # get the sentiment score and the close price of the stock from input file
    sentiment = read_csv(sentimentScore, header=0, index_col =0).values
    stockPrice = read_csv(stockPrice, header = 0, index_col = 0).values
    print(stockPrice)
    # transfer all data into float32
    sentiment = sentiment.astype('float32')
    stockPrice = stockPrice.astype('float32')
    #normalize features
    sentimentS = scaler.fit_transform(sentiment)
    stockPriceS = scaler.fit_transform(stockPrice)
    
    
    # build a supervised learning dataset
    
    score = DataFrame(sentimentS, index = [i for i in range(0, len(sentimentS))])
    price = DataFrame(stockPriceS, index = [i for i in range(0, len(stockPriceS))])
    Tprice = price.shift(-1);
    
    for i in range(0,Tprice.size,1):
        if (Tprice.at[i,0] - price.at[i,0]>0):
           Tprice.at[i,0] = 1
        else:
            Tprice.at[i,0] = 0
    
    cols, names = list(), list()
    
    cols.append(score)
    cols.append(price)
    cols.append(Tprice)
    
    names.append('score')
    names.append('price')
    names.append('trend')
    
    #put names and the data in cols together
    result = concat(cols, axis=1)
    result.columns = names
    
    return result

# load dataset transfer it into a supervised learning data
scaler = MinMaxScaler(feature_range=(0,1))
dataSet = sentiment_stockPrice_series('1.csv', '2.csv',scaler)
values = dataSet.values
 
print('dataset')
print(values)
# split into train and test sets

n_train_days = 15
train = values[:n_train_days, :]
test = values[n_train_days:, :]

# split into input and outputs
train_X, train_y = train[:, :1], train[:, 2]
test_X, test_y = test[:, :1], test[:, 2]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, recurrent_activation='sigmoid',input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='sgd')

# fit network
history = model.fit(train_X, train_y, epochs=600, batch_size=5, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)