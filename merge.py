
import os

import numpy as np
import tensorflow
from keras import Model
from keras.layers import concatenate, GlobalAveragePooling1D, Dense, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2

import loadChain
import similarDays
import targetWeather
import weatherChain
from dataset import prepare_data_set
import keras.backend as K

window_size = 78
lstm_units = 36
dense_units = 72
batch_size = 72
init_mode = 'normal'
last_layer_activation = 'linear'
middle_layer_activation = 'elu'
optimizer = Adam
learn_rate = 0.01

np.random.seed(1120)
tensorflow.random.set_seed(1120)


def loss(y_true, y_pred):
    return K.max(K.square(y_true - y_pred))


def merge():
    loadChainModel = loadChain.train()
    targetWeatherModel = targetWeather.train()
    weatherChainModel = weatherChain.train()
    similarDaysModel = similarDays.train()

    loadChainModel.trainable = False
    targetWeatherModel.trainable = False
    weatherChainModel.trainable = False
    similarDaysModel.trainable = False

    merged = concatenate([loadChainModel.output,
                          targetWeatherModel.output,
                          weatherChainModel.output,
                          similarDaysModel.output],
                         axis=1)
    merged = GlobalAveragePooling1D()(merged)
    merged = Reshape((1, 24))(merged)
    merged = Dense(24, activation='elu')(merged)
    merged = Dense(24, kernel_regularizer=l2(0.01))(merged)
    model = Model(inputs=[loadChainModel.input, targetWeatherModel.input,
                          weatherChainModel.input, similarDaysModel.input],
                  outputs=[merged])
    model.compile(optimizer='adam', loss=loss, metrics=['acc', 'mape'])
    return model


finalModel = merge()
train_x, train_y, eval_x, eval_y, test_x, test_y = prepare_data_set(window_size, batch_size)
if os.path.exists('merge.h5'):
    finalModel.load_weights('merge.h5')
else:
    finalModel.fit(x=[train_x for i in range(0, 4)], y=train_y,
                   validation_data=[[eval_x for i in range(0, 4)], eval_y],
                   shuffle=True, epochs=1000, batch_size=batch_size)
    finalModel.save('merge.h5')
y_pred = finalModel.predict([test_x for i in range(0, 4)], batch_size=batch_size)
for number in y_pred[y_pred.shape[0] - 1, 0, ]:
    print(number)
