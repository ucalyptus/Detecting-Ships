import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks

def design():
    np.random.seed(42)

    # network design
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    return model

def train(X_train, y_train):
    
    model = design()

    # optimization setup
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    # training
    model.fit(
        X_train, 
        y_train,
        batch_size=32,
        epochs=18,
        validation_split=0.2,
        shuffle=True,
        verbose=2)

    return model
