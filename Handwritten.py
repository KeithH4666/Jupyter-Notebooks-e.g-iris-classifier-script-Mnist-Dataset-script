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

## Function to build the model, need to have a folder called dataset on same level as the script
## .gz files must be located in here
def buildModel(filename):

    #f = open(filename)

    ## Build model , adapted from notebook 3 + 4
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

    ## Build the graph, focusing on the metrics of accuracy ##
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with gzip.open('dataset/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()
    
    ##    all images in the training set have an range from 0-1
    ##    and not from 0-255 so we divide our flatten images
    ##    (a one dimensional vector with our 784 pixels)
    ##    to use the same 0-1 based range
    
    sess = tf.Session()

    newImage = tf.image.decode_png(tf.read_file(filename), channels = 1)
    image_resized = tf.image.resize_images(newImage, [28, 28])

    image_resized = sess.run(image_resized)
    #print(image_resized)

    type(image_resized)

    image_resized = np.asarray(image_resized, dtype='float')
    image_resized = image_resized[:,:,0]

    plt.imshow(image_resized, cmap='gray')
    plt.show() 

    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)/ 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    # For encoding categorical variables

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)
    inputs = train_img.reshape(60000, 784)

    #model.fit(inputs, outputs, epochs=5, batch_size=100)

    print(train_lbl[0], outputs[0])
    print("This function works")

def saveImages():
    # Variables for Training Image Set
    trainImagesBytes1=16
    trainImagesBytes2=800

    # Variables for Training Labels Set
    trainLabelsBytes1=8
    trainLabelsBytes2=9
    with gzip.open('dataset/t10k-images-idx3-ubyte.gz', 'rb') as file_images:
        image_contents = file_images.read()
    
# Using gzip we just imported, open the zip files contained in our data folder
    with gzip.open('dataset/t10k-labels-idx1-ubyte.gz', 'rb') as file_labels:
        labels_contents = file_labels.read()

# Loop through the images assigning a corresponding label of the the drawn number
    for x in range(10):
        image = ~np.array(list(image_contents[trainImagesBytes1:trainImagesBytes2])).reshape(28,28).astype(np.uint8)
        labels = np.array(list(labels_contents[trainLabelsBytes1:trainLabelsBytes2])).astype(np.uint8)
        
        # Each byte corresponds to a 1 label so increment by 1
        trainLabelsBytes1+=1
        trainLabelsBytes2+=1
        
        # Every 784 bytes corresponds to a 1 image so increment by 784
        trainImagesBytes1+=784
        trainImagesBytes2+=784
        
        # Save the images with the following format
        # E.G train-(0)_[7]
        # This means the image is from the training set, is the first image in the set and the drawn image is a 7
        cv2.imwrite('train-(' + str(x) + ')' + '_' + str(labels) + '.png', image)


saveImages()
buildModel("trainTest.PNG")


