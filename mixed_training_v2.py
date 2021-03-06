import datasets
import models_v2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
import numpy as np
import argparse
import locale
import os
import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch

percentage_test = .25

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", type=str, required=True,
#	help="path to input dataset of house images")
#args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading Category attributes...")
inputPath = "TestData/small_subset_dummy_categories.csv"
df = datasets.load_category_attributes(inputPath)
# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
inputPath ="TestData/small_subset"
images = datasets.load_images(df, inputPath)
#images = images / 255.0

print("[INFO] loading labels...")
labels = datasets.load_labels("TestData/small_subset_labels.csv")
# labels_encoded = []
# print(labels)

testY = labels[:1250]
trainY = labels[1250:]
opt = Adam(lr=1e-3, decay=1e-3 / 200)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=percentage_test, random_state=42)
(trainCategories, testCategories, trainImages, testImages) = split

#maybe change to 1-hot encoding? If so change loss to "categorical_crossentropy"


#created functions which add output layer to the models created in models.py
#makes our life easier because we can then do all 3 types of models to compare

#TODO: make sure that the model values are saved before putting them into the output layer
#So that they can be concatenated in the mixed model
def compile_mlp(hp):
    mlp = models_v2.create_mlp(trainCategories.shape[1])
    output = Dense(len(np.unique(labels)), activation="softmax")(mlp.output)
    model = Model(inputs = mlp.input, outputs = output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics = ['accuracy'])
    return model


def compile_cnn(hp):
    cnn = models_v2.create_cnn(hp)
    #add output layer
    output = Dense(len(np.unique(labels)), activation="softmax")(cnn.output)
    model = Model(inputs = cnn.input, outputs = output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics = ['accuracy'])
    return model

def compile_mixed_model(hp):
    cnn = models_v2.create_cnn(hp)
    mlp = models_v2.create_mlp(hp)
    combinedInput = concatenate([mlp.output, cnn.output])
    # put mixed input into final relu and then do classification
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(len(np.unique(labels)), activation="softmax")(x)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics = ['accuracy'])
    return model

#tune hyperparameters. Not sure if this will work with the mixed model yet
#use one of three functions above as input
#also add code to save the parameters of best model   
def tune_hyperparams(trainX, trainY, testX, testY, function):
    tuner = RandomSearch(function,
                    objective='val_accuracy',
                    max_trials = 5)
    #search best parameter
    #remember to make a validation set!!!
    tuner.search(trainX,trainY,epochs=3,validation_data=(testX,testY))
    model=tuner.get_best_models(num_models=1)[0]
    #summary of best model
    model.summary()
    

#Runs ResNet50 using only pixel data    
def compile_resNet(trainX, trainY):
    resnet = models_v2.create_resNet()
    output_layer = Dense(len(np.unique(labels)), activation="softmax")(resnet.output)
    model=Model(inputs = resnet.input,outputs=output_layer)
    #STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=1, mode='auto',
            #restore_best_weights=True)
    model.fit(x = trainX, y = trainY, epochs = 1)
    
    model.summary()
                    

#mixed ResNet50 and MLP
def mix(trainX, trainY, trainCategories):
    resnet = models_v2.create_resNet()
    mlp = models_v2.create_mlp(trainCategories.shape[1])
    combinedInput = concatenate([mlp.output, resnet.output])
    # put mixed input into final relu and then do classification
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(len(np.unique(labels)), activation="softmax")(x)
    model = Model(inputs=[mlp.input, resnet.input], outputs=x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics = ['accuracy'])
    model.fit(x = [trainCategories, trainX], y = trainY, epochs = 1)
    model.summary()

#run only resnet
#compile_resNet(trainImages, trainY)  

#run mixed model  
mix(trainImages, trainY, trainCategories)
    
    
# create the MLP and CNN models
#mlp = models.create_mlp(trainCategories.shape[1], regress=False)
#cnn = models.create_cnn(256, 256, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
#combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
#x = Dense(4, activation="relu")(combinedInput)
#x = Dense(len(np.unique(labels)), activation="softmax")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
#model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*

#model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics = ['accuracy'])
# train the model
#print("[INFO] training model...")
#model.fit(
#    x=[trainCategories, trainImages], y=trainY,
#	validation_data=([testCategories, testImages], testY),
#	epochs=1, batch_size=8)
#    
    
	#x=([trainAttrX, trainImagesX], trainY),
	#validation_data=([testAttrX, testImagesX], testY),
	#epochs=200, batch_size=8)
# make predictions on the testing data
#print("[INFO] predicting...")
#preds = model.predict([testCategories, testImages])

#print(preds)