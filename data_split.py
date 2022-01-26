import random
import shutil
import os
from sklearn.feature_extraction import image

from tqdm import tqdm
from pathlib import Path

in_path = Path("./50States10K")
out_path = Path("./50States10KSplit10K")

percentage_train = 0.9

# number of images/states
# set to None for all images
take_subset = True
subset_size = 1000

def split_train_test_val():
    for state in [state for state in in_path.iterdir() if state.is_dir()]:
        images = [image for image in state.iterdir()]
        random.shuffle(images)
        if take_subset:
            images = images[:subset_size]
        print(state)
        for i in tqdm(range(int(percentage_train * len(images)))):
            _, statename, filename = str(images[i]).split("\\")
            dest = out_path / "train" / statename / filename
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(str(images[i]), str(dest))
        for i in range(int(percentage_train * len(images)), len(images)):
            _, statename, filename = str(images[i]).split("\\")
            dest = out_path / "test" / statename / filename
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(str(images[i]), str(dest))

if __name__ == "__main__":
    split_train_test_val()