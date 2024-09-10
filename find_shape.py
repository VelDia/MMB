import os
import cv2
dataset_location = 'voc_dataset'
print(os.listdir(dataset_location))
max_width = 0
max_height = 0
tmp_width = 0
tmp_height = 0
for set_name in ['test', 'train', 'val']:
    set_loc = os.path.join(dataset_location, set_name)
    images_path = os.path.join(set_loc, 'images')
    images_list = sorted(os.listdir(images_path))
    images_path = [os.path.join(images_path, image) for image in images_list if image != '.DS_Store']
    frames = [cv2.imread(image) for image in images_path]
    m = len(frames)
    
    for i,im in enumerate(frames):
        width, height, _ = im.shape
        if tmp_width != width:
            tmp_width = width
            print("change in width:", tmp_width, "num: ", i)
        # if width > max_width:
        #     max_width = width
        #     print("New width:", max_width)
        if tmp_height != height:
            tmp_height = height
            print("change in height:", tmp_height, "num: ", i)
        # if height > max_height:
        #     max_height = height
        #     print("New height:", max_height)
print("Max width:", max_width, "\nMax height:", max_height)        
    # shape = (1348,1454)