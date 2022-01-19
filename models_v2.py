from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#am only experimenting with filter size. Still not done anything with dropout,
# learning rate (I think), batch size, stride, #layers, layer size etc.

#With only experimenting with filter size, the epochs are each taking 20mins!
#This will take so much longer with the other hyperparameters, and a billion times
#longer when using all training data. Am I doing something wrong, or is it normal that tuning 
#takes this long???

#hyperparameter optimisatio (e.g. hp.Int) not implemented here
def create_mlp(dim, hp):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# check to see if the regression node should be added
	#if regress:
	#	model.add(Dense(1, activation="linear"))
	# return our model
	return model

 #hyperparameter optimisation added here only for filter sizes
def create_cnn(hp):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (256, 256, 3)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# Make 3 conv + pooling layers (may want to experiment with them later)
	for i in range(3):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(hp.Int('filter ' + str(i) + ' size',min_value=32,
                    max_value=128, step=16 ), (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
        
        
    # flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)
	# check to see if the regression node should be added
    
	#if regress:
	#	x = Dense(1, activation="linear")(x)
	# construct the CNN
	model = Model(inputs, x)
	# return the CNN
	return model

