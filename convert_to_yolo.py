''' Explaining how the dataset VISO is different from YOLO format

# YOLO data format:
# <object-class> <x-center> <y-center> <width> <height>
# YOLO file folder format
# Main_folder:
#     classes.txt
#     labels:
#         image_filename1.txt
#         image_filename2.txt
#         image_filename3.txt
#         ...
#     images:
#         image_filename1.jpg
#         image_filename2.jpg
#         image_filename3.jpg
#         ...

# VISO MOT data format:
# i.e. gt_centroid.txt
# <image-number> <tracking-number> <x-centroid> <y-centroid> <width> <height> <1> <-1> <-1> <-1>
# VISO File storage folder format:
# mot:
#     object-class:
#         num_video: (i.e. 001, 002, 003)
#             gt:
#                 gt_centroid.txt
#                 gt.txt
#             img:
#                 000001.jpg
#                 000002.jpg
#                 000003.jpg
#                 ...

VISO VOC data format:
VISO File storage folder format:
voc:
    object-class:
        Annotations: 
            000001.xml
            000002.xml
            000003.xml
            ...
        JPEGImages:
            000001.jpg
            000002.jpg
            000003.jpg
            ...
        ImageSets:
            Main:
                test.txt (numbers of the images in separate lines)
                train.txt
                ...

'''

import os
# import cv2 
import shutil
# from pylabel import importer

output_yolo_path = 'yolo_dataset/'
os.makedirs(output_yolo_path, exist_ok=True)
output_images_path = os.path.join(output_yolo_path, 'images')
os.makedirs(output_images_path, exist_ok=True)

dataset_path = 'voc'
dict_folder = {
    'car' : os.path.join(dataset_path, 'car'),
    'plane' : os.path.join(dataset_path, 'plane'),
    'ship' : os.path.join(dataset_path, 'ship'),
    'train' : os.path.join(dataset_path, 'train')
}


''' creating classes.txt file to store object-class from folder-names
classes_txt_path = os.path.join(output_yolo_path, 'classes.txt')

try: 
    with open(classes_txt_path, 'w', newline='') as file:
        for key in dict_folder.keys():
            file.write(key + '\n')
except Exception as e:
    print(f"An error occurred: {e}")    
'''


'''Move images and rename them to unique counter
from VISO dataset '/voc
# creating counter for new name of the files .jpg and .txt
counter = 1
for folder_name, folder_path in dict_folder.items():
    if os.path.exists(folder_path):
        # print(os.listdir(folder_path))
        folder_path = os.path.join(folder_path, 'JPEGImages')
        images_list = sorted(os.listdir(folder_path))
        images_path = [os.path.join(folder_path, path) for path in sorted(os.listdir(folder_path)) if path.endswith('.jpg')]
        for img_path in images_path:
            if os.path.exists(img_path):
                print(img_path)
                try: 
                    new_filename = f"{counter:07d}.jpg"  # Rename files by adding a unique numerical identifier
                    dst_image_path = os.path.join(output_images_path, new_filename)
                    shutil.copy(img_path, dst_image_path)  
                    counter += 1
                except Exception as e:
                    print(f"Error moving {file}: {e}") 
            else:
                print('Path doesn`t exist:', img_path)
    else:
        print('Path doesn`t exist:', folder_path)
'''        
