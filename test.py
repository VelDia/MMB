import os
import cv2
dataset_location = 'voc_dataset'
print(os.listdir(dataset_location))
max_width = 0
max_height = 0
for set_name in ['test', 'train', 'val']:
    set_loc = os.path.join(dataset_location, set_name)
    images_path = os.path.join(set_loc, 'images')
    images_list = sorted(os.listdir(images_path))
    images_path = [os.path.join(images_path, image) for image in images_list if image != '.DS_Store']
    frames = [cv2.imread(image) for image in images_path]
    m = len(frames)
    for im in frames:
        width, height, _ = im.shape
        if width > max_width:
            max_width = width
            print("New width:", max_width)
        if height > max_height:
            max_height = height
            print("New height:", max_height)
        
    shape = (1348,1454)