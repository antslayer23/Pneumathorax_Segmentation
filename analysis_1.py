import os
import pandas as pd
import json
import numpy as np
import pydicom as dicom
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
def main():
    encodings = json.load(open('maskBinary.json', 'r'))

    count = 0
    width = 0
    height = 0

    batch_size = 5000
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
    x = np.zeros((batch_size,width,height,1), dtype = 'uint8')


    #print(y)
    #print(x.shape)



    for i in range(0,batch_size):
        img = dicom.read_file("./train_images/"+files[i]+".dcm")
        x[i,:,:,:] = np.reshape(img.pixel_array, (width,height,1)).astype('uint8')

    print(x)

    yTest = yFull[batch_size:2*batch_size]
    xTest = np.zeros((batch_size,width,height,1))

    for i in range(0,batch_size):
        img = dicom.read_file("./train_images/"+files[i+batch_size]+".dcm")
        xTest[i,:,:,:] = np.reshape(img.pixel_array, (width,height,1)).astype('uint8')



    model_in = keras.layers.Input(shape = (width, height, 1))
    model_0 = layers.Conv2D(32, (7, 7), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_in)

    model_1 = layers.Conv2D(32, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(32, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_0 = layers.Conv2D(64, (1, 1), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_0)

    model_1 = layers.Conv2D(64, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(64, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_0 = layers.Conv2D(128, (1, 1), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_0)

    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_0 = layers.Conv2D(128, (1, 1), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_0)

    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_0 = layers.Conv2D(128, (1, 1), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_0)

    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_0 = layers.Conv2D(128, (1, 1), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_0)

    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_0 = layers.Conv2D(128, (1, 1), padding = "same", kernel_initializer = 'he_uniform', activation = "relu")(model_0)

    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_0)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)
    model_1 = layers.Conv2D(128, (3, 3), padding = "same", kernel_initializer = 'he_uniform')(model_1)
    model_1 = layers.BatchNormalization()(model_1)
    model_1 = layers.ReLU()(model_1)

    model_1_add = layers.Add()([model_0,model_1])
    model_0 =  layers.MaxPooling2D((2,2))(model_1_add)

    model_dense = layers.Flatten()(model_0)
    model_dense = layers.Dense(64, activation ='relu')(model_dense)
    model_dense = layers.Dense(64, activation = 'relu')(model_dense)
    model_out = layers.Dense(2, activation = 'softmax')(model_dense)


    model = keras.Model(model_in,model_out)


    print(model.summary())



    #print(model_0_1.summary())


    #optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001,  epsilon=0.000001)
    #model.compile(optimizer='adam',
    #          loss='sparse_categorical_crossentropy',
    #          metrics=['accuracy'])

    #model.fit(x, y, epochs=10, validation_split = 0.2)

    #test_loss, test_acc = model.evaluate(xTest, yTest)
    #print(test_acc)
    #print(model.predict(xTest))
    #print(yTest)






if __name__ == "__main__":
    main()
