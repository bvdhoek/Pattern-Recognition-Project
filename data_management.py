from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

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

def data_pipeline(csvPath, imgDirectory):
    ds_types = [tf.string] + ([tf.int32]*130)
    ds = tf.data.experimental.CsvDataset(
        filenames=csvPath,
        record_defaults=ds_types,
        header=True,

    )
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=csvPath,
        batch_size=5,
        shuffle_buffer_size=100000,
        shuffle_seed=42,
        num_epochs=1)

    # for k in iter(ds):
    #     print(ds[k].numpy())
    # for x in ds.take(1):
    #     for (k,v) in x:
    #         print(v.numpy())
    #     print(x.element_spec)
        # for key, value in val.items():
        #     print(f"{key:20s}: {value}")
        # for key, value in lab.items():
        #     print(f"{key:20s}: {value}")

    ds = ds.map(reshape_input_dict)
    
    
    
    # # print(ds.element_spec)

def reshape_input_dict(x):
    # parse filename into image
    filename = x["filename"]
    print(filename)
    filepath = tf.strings.join([imgDir,filename])
    print(filepath)
    with tf.compat.v1.Session() as sess:
        filepath = filepath.eval()

    print(filepath)

    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image,channels=3)


    # combine state values into single tensor
    stateL = []
    for s in states:
        stateL.append([x[s]])
    state = tf.concat(stateL,axis=1)
    print(state)

    # combine object values into single tensor
    objectsL = []
    for n in names:
        objectsL.append([x[n]])
    objects = tf.concat(objectsL,axis=1)

    # return tuple of value/label dicts
    # return(({'image':image, 'objects':objects},{'state':state}))
    return(({'objects':objects},{'state':state}))

if __name__ == "__main__":
    # get_data_splits("Data/result_2k.csv","D:/Darknet/50States2K",test_size=0.1, random_state=42)
    data_pipeline("Data/2k.csv", "D:/Darknet/50States2K")