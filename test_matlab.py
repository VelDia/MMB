import matlab.engine

from algorithm1 import calc_difference_mask, remove_false_alarms_one_image, convert_to_binary_mask, morph_operations
from algorithm2 import construct_data_matrix, estimate_rank, fRMC, calc_num_observ_matrix
import numpy as np
import math
import cv2
import os
import csv
import time
start_time = time.time()
dataset_path = 'mot/'
dict_folder = {
    'car' : os.path.join(dataset_path, 'car'),
    'plane' : os.path.join(dataset_path, 'plane'),
    'ship' : os.path.join(dataset_path, 'ship'),
    'train' : os.path.join(dataset_path, 'train'),
}
unknown1 = [1, -1, -1, -1]
counter = 0 
for folder_name, folder_path in dict_folder.items():
    print(folder_name)
    # getting frames
    video_folder = [os.path.join(folder_path, path, 'img') for path in sorted(os.listdir(folder_path)) if path != '.DS_Store']
    for video_name in video_folder:
        counter +=1
        opath_im = os.path.join('output_rois', folder_name, str(counter))
        # saving main directories into variables
        save_preds_p2 = os.path.join(opath_im, 'pred_alg2.txt')
        video_path = os.path.join(opath_im, 'video')
        foreground_path = os.path.join(opath_im, 'foreground')
        background_path = os.path.join(opath_im, 'background')
        mask_alg2_path = os.path.join(opath_im, 'masks_alg2')
        mask_ult_path = os.path.join(opath_im, 'masks_ult')
        # creating absent directories
        os.makedirs(opath_im, exist_ok=True)
        os.makedirs(mask_alg2_path, exist_ok=True)
        os.makedirs(mask_ult_path, exist_ok=True)
        os.makedirs(foreground_path, exist_ok=True)
        os.makedirs(background_path, exist_ok=True)
        os.makedirs(video_path, exist_ok=True)
        video2_path = os.path.join(video_path, 'video_alg2.mp4')

        images_list = sorted(os.listdir(video_name))
        images_path = [os.path.join(video_name, image) for image in images_list if image != '.DS_Store']
        frames = [cv2.imread(image) for image in images_path]
        m = len(frames)
        print(m)

        # # save the video
        width, height, _ = frames[0].shape
        print(frames[0].shape)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video2_path, fourcc, 10, (width, height))

        masks = []
        roiss = []

        with open(save_preds_p2, 'w', newline='') as file:
            writer = csv.writer(file)
            N = calc_num_observ_matrix(frames)
            foreground_images = [0] * m
            for i in range(N, m, N):
                