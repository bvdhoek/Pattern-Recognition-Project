from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
def load_category_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	
	df = pd.read_csv(inputPath, sep=";", header=None)
	# determine (1) the unique zip codes and (2) the number of data
	# points with each zip code
	# return the data frame
	return df

def load_labels(inputPath):
    cols = ["label"]
    df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
	# determine (1) the unique zip codes and (2) the number of data
	# points with each zip code
	# return the data frame
    return df


def load_images(df, inputPath):
    
    images = []
    
    for folder_name in os.listdir(inputPath):
        print("Currently loading images from: " + folder_name)
        folder_path = os.path.join(inputPath, folder_name)
        for image_name in os.listdir(folder_path):
            
            image = cv2.imread(os.path.join(folder_path, image_name))
            
            images.append(image)
        
    return np.array(images)
