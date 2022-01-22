from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from data_management import get_data_splits
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

def resNet():
    input_tensor = Input(shape=(256, 256, 3))
    
    base_model = ResNet50(
        include_top=False, weights=None, input_tensor=input_tensor,
        input_shape=None)
    
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    model = Model(input_tensor, x)
    return model

if __name__ == "__main__":
    # Get data splits
    data_path = "Data/result_10k.csv"
    img_directory = "D:/Darknet/50States10K"
    validation_size = 0.1
    random_state = 42
    read_from_pickle = False
    pickle_path = "10k"

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


    # optimiser
    # Maybe we should use AdamW instead?
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
    model = Model(inputs=resnet.input, output=output_layer)

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
                        batch_size=64,
                        callbacks=callbacks
                       )

    # print summary
    print(model.summary())

    # save model
    model.save("ResNetModel")