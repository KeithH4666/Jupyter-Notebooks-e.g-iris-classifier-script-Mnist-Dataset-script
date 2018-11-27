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

    ############################# TEST DATA ################################

    with gzip.open('dataset/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()

    with gzip.open('dataset/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()
    
    test_img =  np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
    test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

    #########################################################################


    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)
    inputs = train_img.reshape(60000, 784)

    model.fit(inputs, outputs, epochs=3, batch_size=100)

    #print(encoder.inverse_transform(model.predict(test_img[14:15])))
    print(encoder.inverse_transform(model.predict(imageFile)))

def convertImage(imagefile):

    im = Image.open(imagefile).convert('L')

    tv = list(im.getdata())

    tv = [(255 - x) * 1.0 / 255.0 for x in tv]

    buildModel(tv)
###
    #sess = tf.Session()

    #newImage = tf.image.decode_png(tf.read_file(imagefile), channels = 1)
    #image_resized = tf.image.resize_images(newImage, [28, 28])

    #image_resized = sess.run(image_resized)

    #image_resized = np.asarray(image_resized, dtype='float')
    #image_resized = image_resized[:,:,0]

    #image_resized = image_resized.reshape(1, 784)

    #image_resized = np.array(image_resized).astype(np.uint8)/ 255.0

    #buildModel(image_resized)

    #print("Image converted")

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


##saveImages()
convertImage("train-(9)_[9].PNG")
#buildModel()


