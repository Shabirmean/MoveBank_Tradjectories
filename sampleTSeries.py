import numpy
import matplotlib.pyplot as plt
import pandas
import math
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(1)

# load the dataset
dataframe = pandas.read_csv('./tortsm.csv', usecols=[2,3], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
dataset = dataset[0:1000]

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 3

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 2))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 2))

trainY = numpy.reshape(trainY, (trainY.shape[0],2))
testY = numpy.reshape(testY, (testY.shape[0],2))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_dim=2))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=10, batch_size=1, verbose=2)

def randError(l):
    return random.random()*l*0.005

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
'''
testPredict = []
for i in range(len(testX)):
    pred = model.predict(numpy.reshape(testX[i], (1, look_back, 2)))
    testPredict.append(pred[0])
    for y in range(1,look_back+1):
        if i < len(testX)-y:
            testX[i+y][look_back-y][0] = pred[0][0]
            testX[i+y][look_back-y][1] = pred[0][1]
'''

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.6f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.6f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset)[:,0], scaler.inverse_transform(dataset)[:,1])
plt.plot(trainPredictPlot[:,0], trainPredictPlot[:,1])
plt.plot(testPredictPlot[:,0], testPredictPlot[:,1])
plt.show()

plt.plot(scaler.inverse_transform(dataset)[:,0])
plt.plot(trainPredictPlot[:,0])
plt.plot(testPredictPlot[:,0])
plt.show()

plt.plot(scaler.inverse_transform(dataset)[:,1])
plt.plot(trainPredictPlot[:,1])
plt.plot(testPredictPlot[:,1])
plt.show()
