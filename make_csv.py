from pathlib import PurePath
from tqdm import tqdm
import json
import csv

objects = [
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

def generate_csv_from_json(jsonPath,csvPath):
    stateDict = dict(zip(states,range(len(states))))
    objectDict = dict(zip(objects,range(len(objects))))


    filepath = jsonPath
    with open(filepath, 'r') as f:
        data = json.load(f)

    rows = []

    for item in tqdm(data):
        row = [0]*131
        pathParts = PurePath(item["filename"]).parts
        
        filepath = pathParts[-2] + '/' + pathParts[-1]
        row[0]=filepath
        
        state = pathParts[-2]
        row[stateDict[state]+1] = 1

        for object in item["objects"]:
            index = objectDict[object["name"]]
            row[index+51] = 1
        
        rows.append(row)

    header = ['filename'] + states + objects
    with open(csvPath, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def generate_csvs_from_json(jsonPath,pathsPath,statesPath,objectsPath):
    stateDict = dict(zip(states,range(len(states))))
    objectDict = dict(zip(objects,range(len(objects))))


    filepath = jsonPath
    with open(filepath, 'r') as f:
        data = json.load(f)

    pathsRows = []
    statesRows = []
    objectsRows = []

    for item in tqdm(data):
        # row = [0]*131
        pathParts = PurePath(item["filename"]).parts
        
        filepath = pathParts[-2] + '/' + pathParts[-1]
        pathsRows.append([filepath])
        
        state = pathParts[-2]
        statesRow = [0]*50
        statesRow[stateDict[state]] = 1
        statesRows.append(statesRow)

        objectsRow = [0]*80
        for object in item["objects"]:
            index = objectDict[object["name"]]
            objectsRow[index] = 1
        objectsRows.append(objectsRow)

    pathsHeader = ['filename']
    with open(pathsPath, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(pathsHeader)
        writer.writerows(pathsRows)
    
    statesHeader = states
    with open(statesPath, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(statesHeader)
        writer.writerows(statesRows)

    objectsHeader = objects
    with open(objectsPath, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(objectsHeader)
        writer.writerows(objectsRows)

if __name__ == "__main__":
    jsonPath = "Data/result_2k_v4.json"
    pathsPath = "Data/2k_paths.csv"
    statesPath = "Data/2k_states.csv"
    objectsPath = "Data/2k_objects.csv"
    generate_csvs_from_json(jsonPath,pathsPath,statesPath,objectsPath)