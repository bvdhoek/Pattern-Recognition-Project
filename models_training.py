from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from data_management import get_data_splits
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from data_management import data_pipeline
import json
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
tf.keras.optimizers.AdamW = AdamW

def mlp():
    #  with the determined optimal hyper parameters
    num_units = 32
    dropout_rate = 0.1
    hp_batch_size = 100
    hp_optimizer = 'sgd'
    
    input_shape = (80,)
    
    model = Sequential()
    
    #input layer
    model.add(Dense(num_units, input_shape=input_shape, activation='relu',name='objects'))
    
    #dropout layer
    model.add(Dropout(dropout_rate))

    return model


def MLP_HP_TUNING(objectsTrain, objectsValidation, statesTrain, statesValidation):    
    results = []   
    
    
    # Waardes die hij gaat proberen
    
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([100,250]))
    
    print("Starting " + str(len(HP_NUM_UNITS.domain.values) * len((HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value)) * len(HP_OPTIMIZER.domain.values) * len(HP_BATCHSIZE.domain.values)) + " hyper parameter tuning runs")
    
    u = 0
    
    for num_units in HP_NUM_UNITS.domain.values:
      for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for hpoptimizer in HP_OPTIMIZER.domain.values:
            for hpbatch_size in HP_BATCHSIZE.domain.values:
                
                u = u + 1
                
                input_shape = (len(objectsTrain[0]),)
              
                model = Sequential()
              
                #input layer
                model.add(Dense(num_units, input_shape=input_shape, activation='relu'))
              
                #dropout layer
                model.add(Dropout(dropout_rate))
              
                #output layer
                model.add(Dense(len(statesTrain[0]), activation='softmax'))
              
              
                model.compile(loss='categorical_crossentropy', optimizer=hpoptimizer, metrics=['accuracy'])
              
                model.fit(objectsTrain, statesTrain, epochs=50, batch_size=hpbatch_size, verbose=1, validation_split=0.2)
          
                test_results = model.evaluate(objectsValidation, statesValidation, verbose=1)
                line = "Parameters: num_units=" + str(num_units) + ", dropout=" + str(dropout_rate) + ", optimizer=" + str(hpoptimizer) + ", batch_size=" + str(hpbatch_size)
                line2 = f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%'
              
                results.append(line)
                results.append(line2)
                
                print("Finsihed run " + str(u))
                
    for line in results:
        print(line)

def train_MLP():
   
    #  with the determined optimal hyper parameters
    num_units = 32
    dropout_rate = 0.1
    hp_batch_size = 100
    hp_optimizer = 'sgd'
    
    model = mlp()
    
    #output layer
    model.add(Dense(50, activation='softmax', name='state'))
    
    model.compile(loss='categorical_crossentropy', optimizer=hp_optimizer, metrics=['accuracy'])
    
    # model.fit(objectsTrain, statesTrain, epochs=50, batch_size=hp_batch_size, verbose=1, validation_split=0.2)

    # test_results = model.evaluate(objectsValidation, statesValidation, verbose=1)
    # line = "parameters: num_units=" + str(num_units) + ", dropout=" + str(dropout_rate) + ", optimizer=" + str(hp_optimizer) + ", batch_size=" + str(hp_batch_size)
    # print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')


    # Get data splits
    training_data_path = "Data/train_100.csv"
    validation_data_path = "Data/validation_100.csv"
    training_image_directory = "D:/Darknet/training_100"
    validation_image_directory = "D:/Darknet/validation_100"

    training_ds = data_pipeline(training_data_path, training_image_directory, 5, 6000, 42)
    validation_ds = data_pipeline(validation_data_path, validation_image_directory, 5, 6000, 42)
    
    # set up callbacks to keep best epoch
    callbacks = [
                ModelCheckpoint(filepath="MlpModel_100imgs_25epochs_{epoch}",
                                save_best_only=True,
                                monitor="val_accuracy",
                                verbose=1,
                                )
                ]

    # train resNet model
    history = model.fit(x=training_ds,
                        validation_data=validation_ds,
                        epochs=25,
                        callbacks=callbacks
                       )

    # print summary
    print(model.summary())

    # save model
    model.save("MlpModelEnd_100imgs_25epochs")
    
    
def resNet():
    i = Input([None,None,3], dtype=tf.uint8, name='image')
    x = tf.cast(i, tf.float32)
    x = preprocess_input(x)
    core = ResNet50(include_top=False, weights=None,
                    input_shape=None, pooling='avg')
    x = core(x)
    model = Model(inputs=[i],outputs=[x])

    return model

# def compileMLP():
#     banana

def compileResNet():
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
    output_layer = Dense(50, activation="softmax", name='state')(resnet.output)
    model = Model(inputs=resnet.input, outputs=output_layer)

    model.compile(loss=CategoricalCrossentropy(from_logits=False),
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def compileCombined():
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

    # model
    mlp_model = mlp()
    resnet = resNet()
    resnet_out = Dense(32, activation="relu")(resnet.output)
    resnet = Model(inputs=resnet.input, outputs=resnet_out)
    print(mlp_model.output_shape)
    print(resnet.output_shape)
    combinedInput = concatenate([mlp_model.output, resnet.output])
    # x = Dense(4, activation="relu")(combinedInput)
    x = Dense(50, activation="softmax", name='state')(x)
    model = Model(inputs=[mlp_model.input, resnet.input], outputs=x)
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    model.compile(loss=CategoricalCrossentropy(from_logits=False),
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def trainCombined():
    # Get data splits
    training_data_path = "Data/train_100.csv"
    validation_data_path = "Data/validation_100.csv"
    training_image_directory = "D:/Darknet/training_100"
    validation_image_directory = "D:/Darknet/validation_100"

    training_ds = data_pipeline(training_data_path, training_image_directory, 5, 6000, 42)
    validation_ds = data_pipeline(validation_data_path, validation_image_directory, 5, 6000, 42)
    
    # Get model
    model = compileCombined()

    # set up callbacks to keep best epoch
    callbacks = [
                ModelCheckpoint(filepath="CombinedModel_100imgs_25epochs_{epoch}",
                                save_best_only=True,
                                monitor="val_accuracy",
                                verbose=1,
                                )
                ]

    print(training_ds.take(1))
    # train resNet model
    history = model.fit(x=training_ds,
                        validation_data=validation_ds,
                        epochs=25,
                        callbacks=callbacks
                       )

    # print summary
    print(model.summary())

    # save model
    model.save("CombinedModelEnd_100imgs_25epochs")

def trainResNet():
    # Get data splits
    training_data_path = "Data/train_100.csv"
    validation_data_path = "Data/validation_100.csv"
    training_image_directory = "D:/Darknet/training_100"
    validation_image_directory = "D:/Darknet/validation_100"

    training_ds = data_pipeline(training_data_path, training_image_directory, 5, 6000, 42)
    validation_ds = data_pipeline(validation_data_path, validation_image_directory, 5, 6000, 42)
    
    # Get model
    model = compileResNet()

    # set up callbacks to keep best epoch
    callbacks = [
                ModelCheckpoint(filepath="ResNetModel_100imgs_25epochs_{epoch}",
                                save_best_only=True,
                                monitor="val_accuracy",
                                verbose=1,
                                )
                ]

    # train resNet model
    history = model.fit(x=training_ds,
                        validation_data=validation_ds,
                        epochs=25,
                        callbacks=callbacks
                       )

    # print summary
    print(model.summary())

    # save model
    model.save("ResNetModelEnd_100imgs_25epochs")

def evaluate_model(model_path):
    model = keras.models.load_model(model_path)
    ds = data_pipeline("Data/2k.csv","D:/Darknet/50States2K",5,1,42)

    # evaluate, writing results to pickle and human readable json
    evaluate_results = model.evaluate(ds, return_dict=True)
    pickle_name = model_path + "_evaluate.pkl"
    with open(pickle_name, "wb") as pickle_file:
        pickle.dump(evaluate_results, pickle_file)
    json_name = model_path + "_evaluate.json"
    with open(json_name, "w") as json_file:
        json.dump(evaluate_results,json_file)
    
    print("done evaluating")


def predict_results_resnet(model_path):
    print("predicting")
    model = keras.models.load_model(model_path)
    
    ds = data_pipeline("Data/2k.csv","D:/Darknet/50States2K",5,1,42)
    
    states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
    predictions = []
    for d in tqdm(ds):
        predictions += list(model.predict(d[0]["image"]))
    predictions = np.array(predictions)
    
    df = pd.DataFrame(predictions, columns=states)
    predictions_name = model_path+"_predictions.csv"
    df.to_csv(predictions_name)
    print("done predicting")


def predict_results_combined(model_path):
    print("predicting")
    model = keras.models.load_model(model_path)
    
    ds = data_pipeline("Data/2k.csv","D:/Darknet/50States2K",5,1,42)
    
    states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
    predictions = []
    for d in tqdm(ds):
        predictions += list(model.predict([d[0]["objects_input"], d[0]["image"]]))
    predictions = np.array(predictions)
    
    df = pd.DataFrame(predictions, columns=states)
    predictions_name = model_path+"_predictions.csv"
    df.to_csv(predictions_name)
    print("done predicting")

def predict_results_mlp(model_path):
    print("predicting")
    model = keras.models.load_model(model_path)
    
    ds = data_pipeline("Data/2k.csv","D:/Darknet/50States2K",5,50,42)
    
    states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
    predictions = []
    for d in tqdm(ds):
        predictions += list(model.predict(d[0]["objects_input"]))
    predictions = np.array(predictions)
    
    df = pd.DataFrame(predictions, columns=states)
    predictions_name = model_path+"_predictions.csv"
    df.to_csv(predictions_name)
    print("done predicting")



if __name__ == "__main__":
    # # Get data splits
    # training_data_path = "Data/train.csv"
    # validation_data_path = "Data/validation.csv"
    # training_image_directory = "D:/Darknet/training"
    # validation_image_directory = "D:/Darknet/validation"
    # test_data_path = "Data/2k.csv"
    # test_image_directory = "D:/Darknet/50States2K"

    # # validation_size = 0.1
    # random_state = 42
    # # read_from_pickle = False
    # # pickle_path = "2k"

    # training_ds = data_pipeline(training_data_path, training_image_directory, 64, 50, 42)
    # validation_ds = data_pipeline(validation_data_path, validation_image_directory, 64, 50, 42)

    

    # # splits = get_data_splits(data_path=data_path,
    # #                         img_directory=img_directory,
    # #                         test_size= validation_size,
    # #                         random_state=random_state,
    # #                         read_from_pickle=read_from_pickle,
    # #                         pickle_path=pickle_path
    # #                     )
    # # (imagesTrain, imagesValidation,
    # #  objectsTrain, objectsValidation,
    # #  statesTrain, statesValidation
    # # ) = splits
    # # MLP_HP_TUNING(objectsTrain, objectsValidation, statesTrain, statesValidation)



    # # optimiser
    # opt = AdamW(
    #             weight_decay=0.0001,
    #             learning_rate=0.001,
    #             beta_1=0.9,
    #             beta_2=0.999,
    #             epsilon=1e-07,
    #             amsgrad=False,
    #             name="AdamW"
    #             )
    
    # # ResNet model
    # resnet = resNet()
    # output_layer = Dense(50, activation="softmax", name='state')(resnet.output)
    # model = Model(inputs=resnet.input, outputs=output_layer)

    # model.compile(loss=CategoricalCrossentropy(from_logits=False),
    #               optimizer=opt,
    #               metrics=['accuracy'])

    # # set up callbacks to keep best epoch
    # callbacks = [
    #             ModelCheckpoint(filepath="ResNetModel_{epoch}",
    #                             save_best_only=True,
    #                             monitor="val_loss",
    #                             verbose=1,
    #                             )
    #             ]

    # # train resNet model
    # history = model.fit(x=training_ds,
    #                     validation_data=validation_ds,
    #                     epochs=1,
    #                     callbacks=callbacks
    #                    )

    # # print summary
    # print(model.summary())

    # # save model
    # model.save("ResNetModel")

    # mlp1=mlp()

    # test_ds = data_pipeline(test_data_path, test_image_directory, 64, 50, 42)

    # model = keras.models.load_model('ResNetModel')
    # model.predict()

    # print("training combined")
    # trainCombined()


    # print("training mlp")
    # train_MLP()
    # print("training resnet")
    # trainResNet()

    print("evaluating mlp")
    evaluate_model('MlpModel_100imgs_25epochs_15')
    print("evaluating ResNet")
    evaluate_model('ResNetModel_100imgs_25epochs_13')

    print("Predicting mlp")
    predict_results_mlp('MlpModel_100imgs_25epochs_15')
    print("Predicting ResNet")
    predict_results_resnet('ResNetModel_100imgs_25epochs_13')