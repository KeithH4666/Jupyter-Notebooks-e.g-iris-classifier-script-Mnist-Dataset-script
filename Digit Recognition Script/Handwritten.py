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
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

## Function to build the model, need to have a folder called dataset on same level as the script
## .gz files must be located in here

## Canvas all taken from https://www.youtube.com/watch?v=OdDCsxfI8S0
width = 200
height = 200
center = height//2
white = (255, 255, 255)
green = (0,128,0)

def save():
    filename = "image.png"
    image1.save(filename)

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

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

def print_menu():       
    print("-" , "Welcome to Keiths Digit Recognition Script" , 30 * "-")
    print("A. Select your own image 1")
    print("B. Draw you digit")
    print("C. Exit")
   
  
loop=True      
  
while loop:          ## While loop which will keep going until loop = False
    print_menu()    ## Displays menu
    choice = input("Enter your choice [A-C]: ")
    print(choice)
     
    if choice == 'A':     
        userInput = input("Please enter file name/path: ")
        convertImage(userInput)
    elif choice=='B':
        # Canvas taken from https://www.youtube.com/watch?v=OdDCsxfI8S0
        print("Creating canvas (X canvas off when finished and select option one and enter 'image.png')")
        root = tk.Tk()

        # Tkinter create a canvas to draw on
        cv = tk.Canvas(root, width=width, height=height, bg='white')
        cv.pack()

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        image1 = PIL.Image.new("RGB", (width, height), white)
        draw = ImageDraw.Draw(image1)

        # do the Tkinter canvas drawings (visible)
        # cv.create_line([0, center, width, center], fill='green')

        cv.pack()
        cv.bind("<B1-Motion>", paint)

        # do the PIL image/draw (in memory) drawings
        # draw.line([0, center, width, center], green)

        # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
        # filename = "my_drawing.png"
        # image1.save(filename)
        button=tk.Button(text="save",command=save)
        button.pack()
        root.mainloop()
    elif choice=='C':
        print("Exit")
        
        ## You can add your code or functions here
        loop=False # This will make the while loop to end as not value of loop is set to False
    else:
        # Any integer inputs other than values 1-5 we print an error message
        print("Wrong option selection. Enter any key to try again..")




