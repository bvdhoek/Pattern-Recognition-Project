import random
import shutil
import os
from tracemalloc import start

from tqdm import tqdm
from pathlib import Path
from glob import glob
import csv

in_path = Path("./50States10K")
out_path = Path("./50States10KSplit10K")

input_image_path = Path("D:/Darknet/50States10K")
training_image_path = Path("D:/Darknet/training")
validation_image_Path = Path("D:/Darknet/validation")
csv_input_path = Path("D:/Documents - Storage Drive/!UU/2021-2022/2 Pattern Recognition/Pattern-Recognition-Project/Data/10k.csv")
csv_training_path = Path("D:/Documents - Storage Drive/!UU/2021-2022/2 Pattern Recognition/Pattern-Recognition-Project/Data/train.csv")
csv_validation_path = Path("D:/Documents - Storage Drive/!UU/2021-2022/2 Pattern Recognition/Pattern-Recognition-Project/Data/validation.csv")

percentage_train = 0.9

# number of images/states
# set to None for all images
take_subset = False
subset_size = 1000

def split_train_test_val():
    for state in [state for state in in_path.iterdir() if state.is_dir()]: # iterate over all entries in origin directory, keep only directories
        images = [image for image in state.iterdir()] # finds all files in the subdirectory (TODO: should glob for .jpg)
        random.shuffle(images)
        if take_subset:
            images = images[:subset_size] # take subset of images, why?
        print(state)
        for i in tqdm(range(int(percentage_train * len(images)))): # train split
            _, statename, filename = str(images[i]).split("\\")
            dest = out_path / "train" / statename / filename
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(str(images[i]), str(dest))
        for i in range(int(percentage_train * len(images)), len(images)): # val split
            _, statename, filename = str(images[i]).split("\\")
            dest = out_path / "test" / statename / filename
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(str(images[i]), str(dest))

def split_data_and_csv():
    path = os.path.join(input_image_path,'*/*.jpg')
    images = glob(path)

    num_images = len(images)
    state_count = int(num_images / 50)
    state_training_count = int(state_count/5)
    state_validation_count = int(state_count/50)

    with open(csv_input_path,mode='r',newline='') as csv_input, open(csv_training_path, mode='w',newline='') as csv_training, open(csv_validation_path, mode='w',newline='') as csv_validation:
        input_csv_reader = csv.reader(csv_input,delimiter=',')
        training_csv_writer = csv.writer(csv_training,delimiter=',')
        validation_csv_writer = csv.writer(csv_validation,delimiter=',')

        input_csv_reader = [r for r in input_csv_reader]

        header = input_csv_reader.pop(0)
        print(header)
        training_csv_writer.writerow(header)
        validation_csv_writer.writerow(header)

        for i in tqdm(range(50)):
            starting_index = state_count * i
            # copy training data
            for t in range(state_training_count):
                index = starting_index + t

                row = input_csv_reader[index]
                training_csv_writer.writerow(row)

                image_path = images[index]
                rel_path = os.path.relpath(image_path,input_image_path)
                dest = os.path.join(training_image_path,rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(str(image_path), str(dest))

            # copy validation data
            starting_index = starting_index + state_training_count
            for v in range(state_validation_count):
                index = starting_index + v

                row = input_csv_reader[index]
                validation_csv_writer.writerow(row)

                image_path = images[index]
                rel_path = os.path.relpath(image_path,input_image_path)
                dest = os.path.join(validation_image_Path,rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(str(image_path), str(dest))

if __name__ == "__main__":
    split_data_and_csv()