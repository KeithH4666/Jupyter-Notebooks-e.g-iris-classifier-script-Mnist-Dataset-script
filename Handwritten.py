# Import keras.
import keras as kr
from keras.models import Sequential
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import sklearn.preprocessing as pre
import sys
import gzip
import numpy as np

## Function to build the model, need to have a folder called dataset on same level as the script
## .gz files must be located in here
def buildModel():
# Start a neural network, building it by layers.
    model = kr.models.Sequential()

    # Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

# Build the graph.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with gzip.open('dataset/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()
    
    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)/ 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    # For encoding categorical variables

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    print(train_lbl[0], outputs[0])
    print("This function works")

buildModel()


