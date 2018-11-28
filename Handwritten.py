# Import keras.
import keras as kr
import tensorflow as tf
from keras.models import Sequential
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import sklearn.preprocessing as pre
import sys
import gzip
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw

## Function to build the model, need to have a folder called dataset on same level as the script
## .gz files must be located in here
def buildModel(imageFile):

    model = kr.models.Sequential()

    ## https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
    ## Used this reference for dropout values as I was unsure why this was used 
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with gzip.open('dataset/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    train_img =  np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)/ 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)
    inputs = train_img.reshape(60000, 784)

    # Train the model with our inputs(Images) and outputs (Labels)
    print("Building neural network - May take a few mins!")
    model.fit(inputs, outputs, epochs=3, batch_size=100)

    print("According to my network your number is: ")
    print(encoder.inverse_transform(model.predict(imageFile)))

def convertImage(imagefile):

    ## Really good reference for this : http://www.appstate.edu/~marshallst/GLY3455/lectures/9_Image_Processing.pdf

    ## Convert to greyscale
    im = Image.open(imagefile).convert('L')

    ## Make sure image is resized
    im= im.resize((28, 28), Image.BICUBIC)

    ## Convert to list
    im = list(im.getdata())

    # Currently everything is in bytes 0 - 255 , we want to make this 0-1 
    im = [(255 - x) * 1.0 / 255.0 for x in im]
    
    ## need to reshape for our model, expects an array of length 1-D array of size 784
    im =  np.array(list(im)).reshape(1,784)

    print("Image successfully converted! Sending To model")

    ## Send the ready array to our build model function
    buildModel(im)

######### Menu ###########
print("Welcome to Keiths Digit Recognition Script")
print("------------------------------------------")
userInput = input("Please enter file name/path: ")
print("------------------------------------------")

## Send image to our converter to make the image readable for model
## Ready image gets sent to the buildmodel() function
convertImage(userInput)
print("Thanks for using my program!, RE-run to try again.")


