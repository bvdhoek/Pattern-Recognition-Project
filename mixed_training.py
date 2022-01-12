import datasets
import models
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

percentage_test = .25


# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", type=str, required=True,
#	help="path to input dataset of house images")
#args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading Category attributes...")
inputPath = "D:\School\Master\Vakken\MPR\Project\Data\small_subset_dummy_categories.csv"
df = datasets.load_category_attributes(inputPath)
# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
inputPath ="D:/School/Master/Vakken/MPR/Project/Data/50States2K_test/small_subset"
images = datasets.load_images(df, inputPath)
#images = images / 255.0

print("[INFO] loading labels...")
labels = datasets.load_labels("D:\School\Master\Vakken\MPR\Project\Data\small_subset_labels.csv")

testY = labels[:1250]
trainY = labels[1250:]

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=percentage_test, random_state=42)
(trainCategories, testCategories, trainImages, testImages) = split



#x=np.asarray([trainAttrX, trainImagesX]).astype('float32')



# create the MLP and CNN models
mlp = models.create_mlp(trainCategories.shape[1], regress=False)
cnn = models.create_cnn(256, 256, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
# train the model
print("[INFO] training model...")
model.fit(
    x=[trainCategories, trainImages], y=trainY,
	validation_data=([testCategories, testImages], testY),
	epochs=1, batch_size=8)
    
    
	#x=([trainAttrX, trainImagesX], trainY),
	#validation_data=([testAttrX, testImagesX], testY),
	#epochs=200, batch_size=8)
# make predictions on the testing data
print("[INFO] predicting...")
preds = model.predict([testCategories, testImages])

print(preds)