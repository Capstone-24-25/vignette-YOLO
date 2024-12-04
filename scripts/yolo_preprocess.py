import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

# Generates array of all class labels
parent_folder = "../data/crop/"
classes = np.array([folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))])

# Aggregating csv and jpg paths
csv_paths = glob.glob("../data/dataset/*.csv")
jpg_paths = glob.glob("../data/dataset/*.jpg")
csv_paths.sort()
jpg_paths.sort()

# Checks to see if csv contains valid class label
def check_valid_csv(csv_path):
	df = pd.read_csv(csv_path)
	return all(df['class'].isin(classes))

valid_csv_paths = []
valid_jpg_paths = []

for csv_path, jpg_path in tqdm(zip(csv_paths, jpg_paths)):
	if check_valid_csv(csv_path):
		valid_csv_paths.append(csv_path)
		valid_jpg_paths.append(jpg_path)

num_valid = len(valid_csv_paths)
print(f'Valid Paths: {num_valid}')

# Formats data into train/test/validation folders
for i, (csv_path, jpg_path) in enumerate(tqdm(zip(valid_csv_paths, valid_jpg_paths), total = num_valid)):
	annotations = np.array(pd.read_csv(csv_path))
	if i < int(len(valid_csv_paths) * 0.8):   # 80% train
		img_folder = "../data/ultralytics/train/images/"
		label_folder = "../data/ultralytics/train/labels/"
	elif i < int(len(valid_csv_paths) * 0.9): # 10% test
		img_folder = "../data/ultralytics/test/images/"
		label_folder = "../data/ultralytics/test/labels/"
	else:                                     # 10% valid
		img_folder = "../data/ultralytics/valid/images/"
		label_folder = "../data/ultralytics/valid/labels/"

	shutil.copy(jpg_path, img_folder + os.path.basename(jpg_path))
	txt_file_path = label_folder + os.path.basename(csv_path)[:-4] + '.txt'

	with open(txt_file_path, 'w') as f:
		for annotation in annotations:
			width = annotation[1]
			height = annotation[2]
			class_name = annotation[3]
			xmin = annotation[4]
			ymin = annotation[5]
			xmax = annotation[6]
			ymax = annotation[7]
			x_center = 0.5 * (xmin + xmax) / width
			y_center = 0.5 * (ymin + ymax) / height
			b_width = (xmax - xmin) / width
			b_height = (ymax - ymin) / height
			class_num = np.where(classes == class_name)[0][0]
			f.write(f'{class_num} {x_center} {y_center} {b_width} {b_height}\n')

# YAML file for ultralytics model
classes_str = ', '.join([f'"{c}"' for c in classes])
with open('../data/ultralytics/mad.yaml', 'w') as f:
    yaml_string = f'train: ../data/ultralytics/train\n' \
                  f'val: ../data/ultralytics/valid\n' \
                  f'test: ../data/ultralytics/test\n' \
                  f'nc: {len(classes)}\n' \
                  f'names: [{classes_str}]'
    f.write(yaml_string)