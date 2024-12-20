import csv
import os
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
from random import shuffle, seed
from ultralytics import YOLO

classes = np.array(["A10", "A400M", "AG600", "AH64", "An124", "An22", "An225", "An72", "AV8B", "B1", "B2", "B21", "B52", "Be200", "C130", "C17", "C2", "C390", "C5", "CH47", "CL415", "E2", "E7", "EF2000", "F117", "F14", "F15", "F16", "F18", "F22", "F35", "F4", "H6", "J10", "J20", "JAS39", "JF17", "JH7", "Ka27", "Ka52", "KC135", "KF21", "KJ600", "Mi24", "Mi26", "Mi28", "Mig29", "Mig31", "Mirage2000", "MQ9", "P3", "Rafale", "RQ4", "SR71", "Su24", "Su25", "Su34", "Su57", "TB001", "TB2", "Tornado", "Tu160", "Tu22M", "Tu95", "U2", "UH60", "US2", "V22", "Vulcan", "WZ7", "XB70", "Y20", "YF23", "Z19"])

def csv_to_txt(csv_path, output_path):
    """
	Ultralytics requires .txt files to be paired with the .jpg files
	.txt files contain annotations. This function extracts those 
    annotations from the .csv files and converts them into .txt files
	"""
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
    
        for row in reader:
            filename = row['filename']
            width = float(row['width'])
            height = float(row['height'])
            obj_num = np.where(classes == row['class'])[0][0]
            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])

			# YOLO format is class x_center y_center norm_width norm_heigh
            x_center = ((xmin + xmax) / (2 * width))
            y_center = ((ymin + ymax) / (2 * height))
            norm_width = (xmax - xmin) / width
            norm_height = (ymax - ymin) / height

            yolo_format_line = f'{obj_num} {x_center} {y_center} {norm_width} {norm_height}\n'

            txt_filename = os.path.join(output_path, f'{filename}.txt')

            with open(txt_filename, 'a') as txt_file:
                txt_file.write(yolo_format_line)

def train_test_valid(jpgs, csvs, train_ratio, test_ratio, rand_seed=1492):
    """
    Separates dataset into train/test/validation
    Since every jpg file has a corresponding annotations, we
    just have to match them and then shuffle the matched set
    """
    jpgs.sort()
    csvs.sort()
    
    joined_paths = list(zip(jpg_paths, csv_paths))

    seed(rand_seed)
    shuffle(joined_paths)

    train_end = int(len(joined_paths) * train_ratio)
    test_end = train_end + int(len(joined_paths) * test_ratio)

    train_paths = joined_paths[:train_end]
    test_paths = joined_paths[train_end:test_end]
    valid_paths = joined_paths[test_end:]

    return train_paths, test_paths, valid_paths

def data_copy(path_list, target_img_dir, target_label_dir):
    """
    Copies data into ultralytic's format: a path for labels, and a path for images
    """
    for paths in tqdm(path_list):
        jpg = paths[0]
        labels = paths[1]

        csv_to_txt(labels, target_label_dir)
        shutil.copy(jpg, target_img_dir)

######################################
#           Run Preprocess           #
######################################

# Aggregate all the jpg and csv paths in our data
jpg_paths = glob("data/dataset/*.jpg")
csv_paths = glob("data/dataset/*.csv")

# Make directories that the data will be stored in
os.makedirs("data/ultralytics/data/train/images/", exist_ok=True)
os.makedirs("data/ultralytics/data/train/labels/", exist_ok=True)
os.makedirs("data/ultralytics/data/test/images/", exist_ok=True)
os.makedirs("data/ultralytics/data/test/labels/", exist_ok=True)
os.makedirs("data/ultralytics/data/valid/images", exist_ok=True)
os.makedirs("data/ultralytics/data/valid/labels", exist_ok=True)

# Train/test/validation split 80%-10%-10% respectively
train_paths, test_paths, valid_paths = train_test_valid(jpg_paths, csv_paths, 0.8, 0.1)

# Copy data to respective paths
data_copy(test_paths,"data/ultralytics/data/test/images/", "data/ultralytics/data/test/labels/")
data_copy(valid_paths,"data/ultralytics/data/valid/images/", "data/ultralytics/data/valid/labels/")
data_copy(train_paths,"data/ultralytics/data/train/images/", "data/ultralytics/data/train/labels/")

# YOLO requires .yaml file that specifies the directories of the data

# The paths in this file may need to be modified for ultralytics to be able to recognize them
with open("data/ultralytics/data.yaml", 'w') as file:
    class_names = ', '.join(classes)
    data_path = "data/ultralytics/data/"
    train_path = "train/"
    test_path = "test/"
    valid_path = "valid/"
    yaml_str = f'path: {data_path}\n'\
               f'train: {train_path}\n'\
               f'val: {valid_path}\n'\
               f'nc: {len(classes)}\n'\
               f'names: [{class_names}]'
    file.write(yaml_str)