import os
import shutil
import pandas as pd
from glob import glob

# Define paths
base_dir = "unzipped_archive"
crop_dir = os.path.join(base_dir, "crop")
dataset_dir = os.path.join(base_dir, "dataset")
output_dir = "filtered_dataset"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "crop"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "dataset"), exist_ok=True)

# Step 1: Gather classes from `crop` directory
classes = [cls for cls in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, cls))]

# Step 2: Filter `dataset` to include only 3 observations per class
csv_files = sorted(glob(os.path.join(dataset_dir, "*.csv")))
class_counts = {cls: 0 for cls in classes}  # Track counts per class
filtered_files = []  # Store filtered CSV and JPG file paths

for csv_file in csv_files:
    # Read the corresponding CSV
    df = pd.read_csv(csv_file)

    # Check if the class exists in the crop directory
    if "class" in df.columns and df['class'][0] in class_counts:
        cls = df['class'][0]

        # Limit to 3 per class
        if class_counts[cls] < 3:
            filtered_files.append(csv_file)
            class_counts[cls] += 1

# Step 3: Copy filtered dataset files (CSV and JPG) to output
for csv_file in filtered_files:
    # Copy the CSV file
    shutil.copy(csv_file, os.path.join(output_dir, "dataset"))

    # Find and copy the corresponding JPG file
    jpg_file = csv_file.replace(".csv", ".jpg")
    if os.path.exists(jpg_file):
        shutil.copy(jpg_file, os.path.join(output_dir, "dataset"))

# Step 4: Copy `crop` subfolders to output, keeping original images
for cls in classes:
    src_class_dir = os.path.join(crop_dir, cls)
    dst_class_dir = os.path.join(output_dir, "crop", cls)

    # Create class directory in the output
    os.makedirs(dst_class_dir, exist_ok=True)

    # Copy up to 3 images from each class subfolder
    images = sorted(glob(os.path.join(src_class_dir, "*.jpg")))
    for img_file in images[:3]:  # Limit to 3 images
        shutil.copy(img_file, dst_class_dir)

print(f"Filtered dataset created at: {output_dir}")
