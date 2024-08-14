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
import xml.etree.ElementTree as ET

output_yolo_path = 'yolo_voc_dataset/'
os.makedirs(output_yolo_path, exist_ok=True)
output_images_path = os.path.join(output_yolo_path, 'images')
os.makedirs(output_images_path, exist_ok=True)
output_labels_path = os.path.join(output_yolo_path, 'labels')
os.makedirs(output_labels_path, exist_ok=True)

dataset_path = 'voc'
dict_folder = {
    'car' : os.path.join(dataset_path, 'car'),
    'plane' : os.path.join(dataset_path, 'plane'),
    'ship' : os.path.join(dataset_path, 'ship'),
    'train' : os.path.join(dataset_path, 'train')
}
classes = [cls for cls in dict_folder.keys()]

''' creating classes.txt file to store object-class from folder-names'''
classes_txt_path = os.path.join(output_yolo_path, 'classes.txt')

try: 
    with open(classes_txt_path, 'w', newline='') as file:
        for key in dict_folder.keys():
            file.write(key + '\n')
except Exception as e:
    print(f"An error occurred: {e}")    



'''Move images and rename them to unique counter
from VISO dataset '/voc'''  
# creating counter for new name of the files .jpg and .txt
counter = 1
for folder_name, folder_path in dict_folder.items():
    if os.path.exists(folder_path):
        folder_path_img = os.path.join(folder_path, 'JPEGImages')
        folder_path_ann = os.path.join(folder_path, 'Annotations')
        for set_name in ['train', 'val', 'test']:
            ids = open(f'{folder_path}/ImageSets/Main/{set_name}.txt').read().strip().split()
            images_path = [os.path.join(folder_path_img, path+'.jpg') for path in sorted(ids)]
            annot_path = [os.path.join(folder_path_ann, path+'.xml') for path in sorted(ids)]
            # print(ids)
            out_path_img = os.path.join(output_images_path, set_name)
            out_path_annot = os.path.join(output_labels_path, set_name)
            os.makedirs(out_path_img, exist_ok=True)
            os.makedirs(out_path_annot, exist_ok=True)
            for img_path, annot_path in zip(images_path, annot_path):

                # annotations processing
                in_file = open(annot_path)
                out_file = open(os.path.join(out_path_annot, f'{counter:06d}.txt'), 'w')
                tree = ET.parse(in_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = folder_name
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = (b[0] / w, b[1] / w, b[2] / h, b[3] / h)
                    out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
                in_file.close()
                out_file.close()

                # moving images
                new_filename = f"{counter:06d}.jpg"  # Rename files by adding a unique numerical identifier
                dst_image_path = os.path.join(out_path_img, new_filename)
                shutil.copy(img_path, dst_image_path)  

                # increase counter
                counter += 1
    else:
        print('Path doesn`t exist:', folder_path)
      


