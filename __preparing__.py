import numpy as np
from keras.utils import np_utils

def prep(input_data, output_data):

    n_spectrum = 3 # color chanel (RGB)
    weight = 80
    height = 80
    X = input_data.reshape([-1, n_spectrum, weight, height])

    # output encoding
    y = np_utils.to_categorical(output_data, 2)

    # shuffle all indexes
    indexes = np.arange(4000)
    np.random.shuffle(indexes)

    X_train = X[indexes].transpose([0,2,3,1])
    y_train = y[indexes]

    # normalization
    X_train = X_train / 255

    return (X_train,y_train)
    