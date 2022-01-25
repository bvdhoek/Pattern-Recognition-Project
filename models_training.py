from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from data_management import get_data_splits
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
import numpy as np
import tensorflow as tf

def train_MLP(objectsTrain, objectsValidation, statesTrain, statesValidation):
   
 
    input_shape = (len(objectsTrain[0]),)
    print(f'Feature shape: {input_shape}')
    
    model = Sequential()
    model.add(Dense(350, input_shape=input_shape, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(len(statesTrain[0]), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(objectsTrain, statesTrain, epochs=10, batch_size=250, verbose=1, validation_split=0.2)

    test_results = model.evaluate(objectsValidation, statesValidation, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    
    
def resNet():
    i = Input([None,None,3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    x = preprocess_input(x)
    core = ResNet50(include_top=False, weights=None,
                    input_shape=None, pooling='avg')
    x = core(x)
    model = Model(inputs=[i],outputs=[x])

    return model

if __name__ == "__main__":
    # Get data splits
    data_path = "Data/result_2k.csv"
    img_directory = "D:/Darknet/50States2K"
    validation_size = 0.1
    random_state = 42
    read_from_pickle = False
    pickle_path = "2k"

    splits = get_data_splits(data_path=data_path,
                             img_directory=img_directory,
                             test_size= validation_size,
                             random_state=random_state,
                             read_from_pickle=read_from_pickle,
                             pickle_path=pickle_path
                            )
    (imagesTrain, imagesValidation,
     objectsTrain, objectsValidation,
     statesTrain, statesValidation
    ) = splits
    
    train_MLP(objectsTrain, objectsValidation,
     statesTrain, statesValidation)



    # optimiser
    opt = AdamW(
                weight_decay=0.0001,
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name="AdamW"
                )
    
    # ResNet model
    resnet = resNet()
    output_layer = Dense(50, activation="softmax")(resnet.output)
    model = Model(inputs=resnet.input, outputs=output_layer)

    model.compile(loss=CategoricalCrossentropy(from_logits=False),
                  optimizer=opt,
                  metrics=['accuracy'])

    # set up callbacks to keep best epoch
    callbacks = [
                ModelCheckpoint(filepath="ResNetModel_{epoch}",
                                save_best_only=True,
                                monitor="val_loss",
                                verbose=1,
                                )
                ]

    # train resNet model
    history = model.fit(x=imagesTrain,
                        y=statesTrain,
                        validation_data=(imagesValidation,statesValidation),
                        epochs=1,
                        batch_size=32,
                        callbacks=callbacks
                       )

    # print summary
    print(model.summary())

    # save model
    model.save("ResNetModel")
    