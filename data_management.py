import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        images.dump(pickle_path)

    ### CREATE TRAIN/TEST SPLITS ###
    return train_test_split(images, objects, states,
                            test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    get_data_splits("Data/labels.csv","D:/Darknet/50States2K",test_size=0.1, random_state=42)