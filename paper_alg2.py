import os
import cv2
# Calculate the number of observation matrices
images_folder = 'mot/VISO_paper/coco/train/test2017/'
image_files = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg')])
frames = [cv2.imread(image_file) for image_file in image_files]
M = len(frames) 
L = 4
f = 10

N = M/(L * f)
# Estimate the current background model based on LRMC (used test_LRMC_paper.py)
background_images_folder = '/home/diana/output_background_images'

