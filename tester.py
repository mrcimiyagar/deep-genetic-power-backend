import os

import matplotlib.pyplot as plt
import numpy as np


class Tester:

    def __init__(self, dir_path):
        self.dir = dir_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def test_svm(self, model, features_batch, test_y, batch_size, svm, transformers):

        for transformer in transformers:
            features_batch = transformer.transform(features_batch)
        y_pred = svm.predict(features_batch)
        print('actual:     ', list(np.int_(test_y[test_y.shape[0] - 1:, 0, ].reshape((24)))))
        print('prediction: ', list(np.int_(y_pred[y_pred.shape[0] - 24:, ])))
        plt.ion()
        plt.plot(list(np.int_(test_y[test_y.shape[0] - 1:, 0, ].reshape((24)))), color='red')
        plt.plot(list(np.int_(y_pred[y_pred.shape[0] - 24:, ])), color='blue')
        plt.savefig(self.dir + '/svm_9_tir.png')
        plt.close()

    def test_neuron(self, model, test_x, test_y, batch_size):
        pass
