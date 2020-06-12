import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer


def prepare_data_set(window_size, batch_size):
    data = pd.read_excel('datasetW.xlsx', sheet_name='input')
    df = pd.DataFrame(data,
                      columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                               'Temperature', 'Humidity', 'WeatherState'])
    weather = np.array(df)
    data = pd.read_excel('datasetW.xlsx', sheet_name='output')
    df = pd.DataFrame(data, columns=['Load'])
    load = np.array(df)

    weather.resize((weather.shape[0], 12))
    temp = []
    temp[:] = weather[:, 0]
    weather[:, 0] = load[:, 0]
    weather[:, 11] = temp[:]
    dataset = np.array(weather)
    dataset.resize((dataset.shape[0], 14))

    for i in range(0, dataset.shape[0]):
        if i < 24:
            dataset[i, 11] = dataset[i, 0]
            dataset[i, 12] = dataset[i, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        if i < 24 * 7:
            dataset[i, 11] = dataset[i - 24, 0]
            dataset[i, 12] = dataset[i, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        if i < 365 * 24:
            dataset[i, 11] = dataset[i - 24, 0]
            dataset[i, 12] = dataset[i - 24 * 7, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        dataset[i, 11] = dataset[i - 24, 0]
        dataset[i, 12] = dataset[i - 24 * 7, 0]
        dataset[i, 13] = dataset[i - 365 * 24, 0]

    dataset = dataset.astype('float32')

    x, y = list(), list()
    for i in range(window_size, dataset.shape[0] - 23):
        temp = list(dataset[i - window_size: i, ].flatten())
        for j in range(0, 24):
            for l in range(1, 14):
                temp.append(dataset[i + j, l])
        x.append(np.array(temp))
        y.append(list(dataset[i: i + 24, 0].flatten()))

    x, y = np.array(x), np.array(y)

    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    x = scaler_minmax.fit_transform(x)

    normalizer = Normalizer()
    x = normalizer.fit_transform(x)

    x = x.reshape((x.shape[0], 1, window_size * 14 + 24 * 13))
    y = y.reshape((y.shape[0], 1, 24))

    train_x, train_y, eval_x, eval_y, testX, testY = \
        np.array(x[0: 200 * 72, ]), np.array(y[0: 200 * 72, ]), \
        np.array(x[200 * 72: 300 * 72, ]), np.array(y[200 * 72: 300 * 72, ]), \
        np.array(x[300 * 72:, ]), np.array(y[300 * 72:, ])

    extra = train_x.shape[0] - (train_x.shape[0] // batch_size) * batch_size
    train_x = train_x[extra:, ]
    train_y = train_y[extra:, ]
    extra = eval_x.shape[0] - (eval_x.shape[0] // batch_size) * batch_size
    eval_x = eval_x[extra:, ]
    eval_y = eval_y[extra:, ]
    extra = testX.shape[0] - (testX.shape[0] // batch_size) * batch_size
    testX = testX[extra:, ]
    testY = testY[extra:, ]

    return train_x, train_y, eval_x, eval_y, testX, testY
