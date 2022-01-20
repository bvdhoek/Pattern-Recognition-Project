import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def get_data_splits(data_path, img_directory,
                    test_size=None, random_state=None,
                    read_from_pickle=False):
    """Retrieves the data in \'data_path\' (as in labels.csv),
    processes it and generates splits.
    Returns (imagesX, imagesY, objectsX, objectsY, statesX, statesY).
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
            images = np.load('images.npy')
        except Exception:
            print("Something went wrong reading images.npy, "
                  "does it exist in your current directory?")
    else:
        base_img_path = img_directory + '/'
        images = []
        for filepath in relative_paths:
            images.append(cv2.imread(base_img_path + filepath))
        images = np.array(images)
        images.dump('images.npy')


    ### CREATE TRAIN/TEST SPLITS ###
    return train_test_split(images, objects, states,
                            test_size=test_size, random_state=random_state)

