import os
import shutil
import random

# Path to the dataset folder
dataset_path = '/home/robin/d/CUB_200_2011/'
classes_txt_path = os.path.join(dataset_path, 'classes.txt')

# # Read class names from classes.txt
# with open(classes_txt_path, 'r') as f:
#     classes_info = [line.strip().split(' ') for line in f.readlines()]
# class_names = [class_info[1] for class_info in classes_info]

# # Randomly select 100 classes
# random.seed(42)  # Set seed for reproducibility
# selected_classes = random.sample(class_names, 100)

# # Create in_out_splits.txt
# in_out_splits_txt_path = 'in_out_splits.txt'
# with open(in_out_splits_txt_path, 'w') as f:
#     for class_name in class_names:
#         is_in = 1 if class_name in selected_classes else 0
#         f.write(f'{class_name} {is_in}\n')

in_out_splits_txt_path = 'in_out_splits.txt'

with open(in_out_splits_txt_path, 'r') as f:
    lines = f.readlines()

selected_classes = []
for line in lines:
    class_name, is_in = line.strip().split(' ')
    if is_in == '1':
        selected_classes.append(class_name)

# Define source and destination folders
src_folders = ['test_cropped', 'train_cropped', 'train_cropped_augmented']
dst_folders_in = ['test_cropped_in', 'train_cropped_in', 'train_cropped_augmented_in']
dst_folders_out = ['test_cropped_out', 'train_cropped_out', 'train_cropped_augmented_out']

# Copy relevant folders
for src, dst_in, dst_out in zip(src_folders, dst_folders_in, dst_folders_out):
    src_path = os.path.join(dataset_path, src)
    
    for class_name in os.listdir(src_path):
        src_class_path = os.path.join(src_path, class_name)
        is_in = class_name in selected_classes

        if is_in:
            dst_class_path = os.path.join(dataset_path, dst_in, class_name)
        else:
            dst_class_path = os.path.join(dataset_path, dst_out, class_name)

        os.makedirs(dst_class_path, exist_ok=True)
        for file_name in os.listdir(src_class_path):
            src_file_path = os.path.join(src_class_path, file_name)
            dst_file_path = os.path.join(dst_class_path, file_name)
            shutil.copy(src_file_path, dst_file_path)

print("Folders copied.")