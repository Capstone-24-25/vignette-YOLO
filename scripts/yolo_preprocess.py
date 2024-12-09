import csv
import os
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
from random import shuffle, seed
from ultralytics import YOLO

parent_dir = "data/crop"
classes = np.array([folder for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))])

def csv_to_txt(csv_path, output_path):
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

            x_center = ((xmin + xmax) / (2 * width))
            y_center = ((ymin + ymax) / (2 * height))
            norm_width = (xmax - xmin) / width
            norm_height = (ymax - ymin) / height

            yolo_format_line = f'{obj_num} {x_center} {y_center} {norm_width} {norm_height}\n'

            txt_filename = os.path.join(output_path, f'{filename}.txt')

            with open(txt_filename, 'a') as txt_file:
                txt_file.write(yolo_format_line)

def train_test_valid(jpgs, csvs, train_ratio, test_ratio, rand_seed=1492):
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
    for paths in tqdm(path_list):
        jpg = paths[0]
        labels = paths[1]

        csv_to_txt(labels, target_label_dir)
        shutil.copy(jpg, target_img_dir)

######################################
#           Run Preprocess           #
######################################

jpg_paths = glob("data/dataset/*.jpg")
csv_paths = glob("data/dataset/*.csv")

os.makedirs("data/ultralytics/data/train/images/", exist_ok=True)
os.makedirs("data/ultralytics/data/train/labels/", exist_ok=True)
os.makedirs("data/ultralytics/data/test/images/", exist_ok=True)
os.makedirs("data/ultralytics/data/test/labels/", exist_ok=True)
os.makedirs("data/ultralytics/data/valid/images", exist_ok=True)
os.makedirs("data/ultralytics/data/valid/labels", exist_ok=True)

train_paths, test_paths, valid_paths = train_test_valid(jpg_paths, csv_paths, 0.8, 0.1)

data_copy(test_paths,"data/ultralytics/data/test/images/", "data/ultralytics/data/test/labels/")
data_copy(valid_paths,"data/ultralytics/data/valid/images/", "data/ultralytics/data/valid/labels/")
data_copy(train_paths,"data/ultralytics/data/train/images/", "data/ultralytics/data/train/labels/")

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