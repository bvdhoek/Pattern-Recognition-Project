import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

imgDir = tf.constant("D:/Darknet/50States2K/")

names = [
        'person',
        'bicycle',
        'car',
        'motorbike',
        'aeroplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'sofa',
        'pottedplant',
        'bed',
        'diningtable',
        'toilet',
        'tvmonitor',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush']

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming"]

def get_data_splits(data_path, img_directory,
                    test_size=None, random_state=None,
                    read_from_pickle=False, pickle_path='images.npy'):
    """Retrieves the data in \'data_path\' (as in labels.csv),
    processes it and generates splits.
    Returns (imagesTrain, imagesValidation, objectsTrain, objectsValidation, 
    statesTrain, statesValidation).
    Both objects and states are one-hot encoded in numpy arrays and
    images consists of numpy arrays with image data."""
    ### RETRIEVE DATA ###
    # read all data
    df = pd.read_csv(data_path,sep=',', header=0, index_col=0)

    # separate objects
    objects = df.loc[:, ~df.columns.isin(['state', 'filename'])]
    # one-hot encode states
    states = pd.get_dummies(df.state)
    # generate filepaths
    df['relativepath'] = df['state'] + '/' + df['filename']
    relative_paths = df.loc[:,'relativepath']
    

    ### PROCESS DATA ###
    # turn object input and state targets into numpy arrays
    objects = objects.to_numpy()
    states = states.to_numpy()
    # read images
    if read_from_pickle:
        try:
            images = np.load(pickle_path)
        except Exception:
            print("Something went wrong opening the pickle, "
                  "does it exist at pickle_path?")
    else:
        base_img_path = img_directory + '/'
        images = []
        for filepath in tqdm(relative_paths):
            images.append(cv2.imread(base_img_path + filepath))
        images = np.array(images)
        # images.dump(pickle_path)

    ### CREATE TRAIN/TEST SPLITS ###
    return train_test_split(images, objects, states,
                            test_size=test_size, random_state=random_state)


def data_pipeline(csvPath, imgDirectory, batchSize, bufferSize, seed):
    # Read CSV
    col_numbers = [i for i in range(51,131)]
    csv_ds = tf.data.experimental.make_csv_dataset(
        file_pattern=csvPath,
        select_columns=col_numbers,
        batch_size=batchSize,
        shuffle=False,
        num_epochs=1
        )
    
    # Fetch Images
    img_ds = image_dataset_from_directory(
        imgDirectory,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batchSize,
        image_size=(256, 256),
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    # Combine Datasets
    ds = tf.data.Dataset.zip((img_ds,csv_ds))
    # for x in ds.take(1):
    #     print(type(x))
    #     print(x)

    # Translate to tuple of dicts
    ds = ds.map(reshape_input_tuple)
    # print("final ds")
    # for x in ds.take(1):
    #     print(x)

    # Shuffle Data
    ds = ds.shuffle(
        buffer_size=bufferSize,
        seed=seed,
        reshuffle_each_iteration=False
    )

    # Prefetch Data
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # print("prefetch")
    # for x in ds.take(1):
    #     print(x)

    return(ds)

# def is_test(x, y):
#     return x % 10 == 0

# def is_train(x, y):
#     return not is_test(x, y)

def reshape_input_tuple(image_state,objects):
    """Maybe we should change objects to floats"""
    (image,state) = image_state

    # combine object values into single tensor
    objectsL = []
    for n in names:
        objectsL.append([objects[n]])
    objects = tf.concat(objectsL,axis=1)

    # return tuple of value/label dicts
    return(({'image':image,'objects_input':objects},{'state':state}))

if __name__ == "__main__":
    data_pipeline("Data/2k.csv", "D:/Darknet/50States2K", 1, 10000, 42)