import os
import pandas as pd
import json
import numpy as np
import pydicom as dicom
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
def main():
    encodings = json.load(open('maskBinary.json', 'r'))

    count = 0
    width = 0
    height = 0

    batch_size = 500
    regularization = 0.000001

    for file in os.listdir("./train_images"):
        if(count != 0):
            break

        img = dicom.read_file("./train_images/"+file)
        width = img.pixel_array.shape[0]
        height = img.pixel_array.shape[1]


        count = 1

    yFull = np.zeros(len(encodings))
    files = np.empty(len(encodings), dtype='object')

    index = 0
    for i in encodings:
        yFull[index] = int(encodings[i])
        #print(i)
        files[index] = i
        index = index+1


    y = yFull[0:batch_size]
    x = np.zeros((batch_size,width,height,1))
    #print(y)
    #print(x.shape)



    for i in range(0,batch_size):
        img = dicom.read_file("./train_images/"+files[i]+".dcm")
        x[i,:,:,:] = np.reshape(img.pixel_array, (width,height,1)).astype(float)*2/255.0-1.0



    yTest = yFull[batch_size:2*batch_size]
    xTest = np.zeros((batch_size,width,height,1))

    for i in range(0,batch_size):
        img = dicom.read_file("./train_images/"+files[i+batch_size]+".dcm")
        xTest[i,:,:,:] = np.reshape(img.pixel_array, (width,height,1)).astype(float)/255.0


    model = models.Sequential()
    model.add(layers.Conv2D(4, (3, 3), input_shape=(width, height, 1),kernel_initializer = 'he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(4, (3, 3), activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer=tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3),kernel_initializer = 'he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(4, (3, 3), activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer=tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3),kernel_initializer = 'he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(4, (3, 3), activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer=tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3),kernel_initializer = 'he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(4, (3, 3), activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer=tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3),kernel_initializer = 'he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(4, (3, 3), activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer=tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3),kernel_initializer = 'he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(4, (3, 3), activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer=tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer = tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.Dense(64, activation='elu',kernel_initializer = 'he_uniform', kernel_regularizer = tf.keras.regularizers.l2(regularization),
                activity_regularizer=tf.keras.regularizers.l1(regularization)))
    model.add(layers.Dense(2, activation='softmax',kernel_initializer = 'he_uniform'))

    print(model.summary())


    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005,  epsilon=0.000001)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x, y, epochs=10, validation_split = 0.2)

    test_loss, test_acc = model.evaluate(xTest, yTest)
    print(test_acc)
    print(model.predict(xTest))
    print(yTest)

    #print(x[:,:,:,0])
    #mlp = MLPClassifier(hidden_layer_sizes = (50,50,50,50), activation = "relu", solver = "adam", random_state = 0, max_iter = 20)
    #mlp.fit(x,y)


    #y = yFull[batch_size:2*batch_size]
    #x = np.zeros((batch_size,width*height))
    #print(y.shape)
    #print(x.shape)



    #index = 0
    #for i in range(batch_size,2*batch_size):
    #    if(index == batch_size):
    #        break
    #    img = dicom.read_file("./train_images/"+files[i]+".dcm")
    #    x[index,:] = img.pixel_array.flatten()
#
#        index = index+1

    #print(x)
    #print(mlp.score(x,y))


if __name__ == "__main__":
    main()
