import os

from keras import Model
from keras.layers import LSTM, Input, Dense, Reshape, Conv1D, Lambda, TimeDistributed, MaxPooling1D, \
    Flatten, GlobalAveragePooling1D, concatenate, Permute
from keras.optimizers import Nadam, Adam

from dataset import prepare_data_set

window_size = 78
lstm_units = 36
dense_units = 72
batch_size = 72
init_mode = 'normal'
last_layer_activation = 'linear'
middle_layer_activation = 'elu'
optimizer = Adam
learn_rate = 0.01

train_x, train_y, eval_x, eval_y, test_x, test_y = prepare_data_set(window_size, batch_size)

kmodel_in = Input(batch_shape=(batch_size, test_x.shape[1], test_x.shape[2]))


def create_model():
    global window_size
    global lstm_units
    global dense_units
    global kmodel_in
    global init_mode
    global last_layer_activation
    global middle_layer_activation

    kmodel1 = Lambda(lambda input_x: input_x[:, :, 0: window_size * 14],
                     output_shape=(1, window_size * 14))(kmodel_in)

    kmodel1 = LSTM(lstm_units, kernel_initializer=init_mode, activation=middle_layer_activation)(kmodel1)
    kmodel1 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel1)
    kmodel1 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel1)
    kmodel1 = Reshape((1, dense_units))(kmodel1)
    kmodel = Dense(24, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel1)

    return kmodel


def create_final_model():
    global optimizer
    global learn_rate

    kmodel = create_model()
    kmodel = Model(inputs=[kmodel_in], outputs=[kmodel])
    kmodel.compile(optimizer='adam', loss='mse', metrics=['acc', 'mape'])
    return kmodel


def train():
    model = create_final_model()
    if os.path.exists('weather-chain.h5'):
        model.load_weights('weather-chain.h5')
    else:
        model.fit(x=train_x, y=train_y[:, :, :], batch_size=batch_size, epochs=100, verbose=1,
                  validation_data=[eval_x, eval_y[:, :, :]],
                  shuffle=False)
        model.save('weather-chain.h5')
    return model
