import os
import cv2
# dataset_location = 'yolo_cars'
# print(os.listdir(dataset_location))
max_width = 0
max_height = 0
tmp_width = 0
tmp_height = 0
# for set_name in ['test', 'train', 'val']:
#     set_loc = os.path.join(dataset_location, set_name)
    # images_path = os.path.join(set_loc, 'images')
images_paths = "yolo_voc_dataset/test/images"
images_list = sorted(os.listdir(images_paths))
images_path = [os.path.join(images_paths, image) for image in images_list if image != '.DS_Store']
frames = [cv2.imread(image) for image in sorted(images_path)]
m = len(frames)
low_path = []
med_path = []
high_path = []

for i,im in zip(images_list, frames):
    width, height, _ = im.shape
    if tmp_width != width:
        tmp_width = width
        print("change in width:", tmp_width, "file: ", i)
    if width in range(220,290):
        low_path.append(i)
    elif width in range(450,520):
        med_path.append(i)
    elif width in range(1000, 1350):
        high_path.append(i)
    if width > max_width:
        max_width = width
    #     print("New width:", max_width)
    if tmp_height != height:
        tmp_height = height
        print("change in height:", tmp_height,"file: ", i)
    if height > max_height:
        max_height = height
    #     print("New height:", max_height)

print("Max width:", max_width, "\nMax height:", max_height)    
print("Num of low:", len(low_path), "\nNum of med:", len(med_path), "\nNum of high:", len(high_path))    
    # shape = (1348,1454)


import shutil

def copy_files(file_list, src_dir, dest_dir):
    """
    Copy files from src_dir to dest_dir.

    :param file_list: List of file names to copy
    :param src_dir: Source directory containing the files
    :param dest_dir: Destination directory where files should be copied
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # Create the destination directory if it doesn't exist

    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)  # Copy file with metadata
        else:
            print(f'{file_name} does not exist in {src_dir}')

# src_dir = images_paths
# dest_dir_low = 'yolo_cars/low/images'
# dest_dir_med = 'yolo_cars/med/images'
# dest_dir_high = 'yolo_cars/high/images'

# copy_files(low_path, src_dir, dest_dir_low)
# copy_files(med_path, src_dir, dest_dir_med)
# copy_files(high_path, src_dir, dest_dir_high)

# src_dir = "yolo_cars/test/labels"
# low_path = [l.replace('.jpg', ".txt") for l in low_path]
# med_path = [l.replace('.jpg', ".txt") for l in med_path]
# high_path = [l.replace('.jpg', ".txt") for l in high_path]

# dest_dir_low = 'yolo_cars/low/labels'
# dest_dir_med = 'yolo_cars/med/labels'
# dest_dir_high = 'yolo_cars/high/labels'

# copy_files(low_path, src_dir, dest_dir_low)
# copy_files(med_path, src_dir, dest_dir_med)
# copy_files(high_path, src_dir, dest_dir_high)

