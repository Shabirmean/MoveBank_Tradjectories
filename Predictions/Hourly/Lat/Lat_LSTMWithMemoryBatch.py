import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from six.moves import cPickle


# save an object as pickle
def savePickleFile(myObject, name):
    f = open(name, 'wb')
    cPickle.dump(myObject, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

# open a pickle object
def openPickleFile(name):
    f = open(name, 'wb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# setup keras callback for early stopping
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto')

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('./tortsm.csv', usecols=[3, 8], engine='python')
full_dataset = dataframe.values
names = dataframe['individual-local-identifier'].astype('str')
full_dataset = full_dataset[:, 0:1].astype('float32')

# Predicting only for one tortoise
for tortoise in ['Connor', 'Isabela', 'Miriam', 'Nathalie', 'Sepp']:
    print("Generating Model for Tortoise %s" % (tortoise))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    name_of_tortoise = tortoise
    dataset = full_dataset[numpy.array(names) == name_of_tortoise]

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(-4, 4))
    dataset = scaler.fit_transform(dataset)

    # dataset = dataset[:1000, :]

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 10
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(50):
        print("Epoch No: ", i)
        model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False, callbacks=[early],
                  validation_split=0.2)
        model.reset_states()

    # save the model related input objects
    f = open('./LatResults/' + name_of_tortoise + '_objects.save', 'wb')
    for obj in [trainX, trainY, testX, testY]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # Save the model generated
    model.save('./LatResults/' + name_of_tortoise + '_model.h5')

    # save the model architecture as a JSON
    json_string = model.to_json()
    savePickleFile(json_string, './LatResults/' + name_of_tortoise + '_json.json')
    print("--------------------------------------------------")
    print("Completed Model for Tortoise %s" % (tortoise))
    print("--------------------------------------------------")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # Load saved objects prior to predictions
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # model = load_model('./LatResults/' + name_of_tortoise + '_model.h5')
    #
    # f = open('./LatResults/' + name_of_tortoise + '_objects.save', 'rb')
    # loaded_objects = []
    # for i in range(4):
    #     loaded_objects.append(cPickle.load(f))
    # f.close()

    # trainX_old = trainX
    # trainY_old = trainY
    # testX_old = testX
    # testY_old = testY

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # make predictions
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # trainPredict = model.predict(trainX, batch_size=batch_size)
    # model.reset_states()
    # testPredict = model.predict(testX, batch_size=batch_size)
    #
    # # invert predictions
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform([trainY])
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform([testY])
    #
    # # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train Score: %.6f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test Score: %.6f RMSE' % (testScore))
    #
    # # shift train predictions for plotting
    # trainPredictPlot = numpy.empty_like(dataset)
    # trainPredictPlot[:, :] = numpy.nan
    # trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    #
    # # shift test predictions for plotting
    # testPredictPlot = numpy.empty_like(dataset)
    # testPredictPlot[:, :] = numpy.nan
    # testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    #
    # # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()