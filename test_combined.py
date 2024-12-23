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
        save_preds_p1 = os.path.join(opath_im, 'pred_alg1.txt')
        save_preds_p2 = os.path.join(opath_im, 'pred_alg2.txt')
        video_path = os.path.join(opath_im, 'video')
        foreground_path = os.path.join(opath_im, 'foreground')
        background_path = os.path.join(opath_im, 'background')
        mask_alg1_path = os.path.join(opath_im, 'masks_alg1')
        mask_alg2_path = os.path.join(opath_im, 'masks_alg2')
        mask_ult_path = os.path.join(opath_im, 'masks_ult')
        # creating absent directories
        os.makedirs(opath_im, exist_ok=True)
        os.makedirs(mask_alg1_path, exist_ok=True)
        os.makedirs(mask_alg2_path, exist_ok=True)
        os.makedirs(mask_ult_path, exist_ok=True)
        os.makedirs(foreground_path, exist_ok=True)
        os.makedirs(background_path, exist_ok=True)
        os.makedirs(video_path, exist_ok=True)
        video1_path = os.path.join(video_path, 'video_alg1.mp4')
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
        video = cv2.VideoWriter(video1_path, fourcc, 10, (width, height))

        masks = []
        roiss = []

        with open(save_preds_p1, 'w', newline='') as file:
            writer = csv.writer(file)

            # Algorithm #1
            for i in range(1, m-2):
                im_path_list = images_path[i:i+3]
                # print(im_path_list)
                image = cv2.imread(im_path_list[0], cv2.IMREAD_COLOR)

                diff, mask = calc_difference_mask(im_path_list)
                masks.append(mask)

                mask = morph_operations(mask)
                result_image, rois, new_mask = remove_false_alarms_one_image(mask, image, i)
                cv2.imwrite(os.path.join(mask_alg1_path, f'mask_{i}.png'), new_mask)
                # cv2.imshow('ROI', result_image)
                # cv2.waitKey(0)

                writ_rois = [roi + unknown1 for roi in rois]
                roiss.append(rois)
                writer.writerows(writ_rois)
                masks.append(new_mask) 
                video.write(result_image)
        video.release()
        cv2.destroyAllWindows()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video2 = cv2.VideoWriter(video2_path, fourcc, 10, (width, height))
        masks2 = []
        roiss2 = []
        # Algorithm 2
        with open(save_preds_p2, 'w', newline='') as file:
            writer2 = csv.writer(file)

            # Calculate the number of observation matrices 
            N = calc_num_observ_matrix(frames)
            # if N > 4:
            #     N = 4
            # print(N)
            # print(len(frames[N-N:N]))
            foreground_images = [0] * m
            for i in range(N, m, N):
                iter = i-N
                V = construct_data_matrix(frames[iter:i])
                rank = estimate_rank(V)
                B = fRMC(V, rank)
                background_images = [B[:, i].reshape(frames[0].shape) for i in range(B.shape[1])]
                # foreground_images = [[None] * shape[1] for _ in range(shape[0])]
                binary_images = []
                for k in range(len(background_images)):#range(N)
                    temp_img = frames[i-N+k] - background_images[k]
                    foreground_images[i-N+k] = temp_img
                    
                    # # B = convert_to_binary_mask(B)
                    print(foreground_images[0].shape)
                    binary_images.append(convert_to_binary_mask(temp_img))
                    # binary_images = [convert_to_binary_mask(fr) for fr in foreground_images[iter:iter+N]]
                morphed_images = [morph_operations(binary_image) for binary_image in binary_images]
                # for j in range(B.shape[1]):
                #     background = B[:, j].reshape(frames[0].shape)
                #     foreground = frames[j] - background

                #     # Convert to binary image
                #     binary_image = convert_to_binary_mask(foreground)

                #     # Perform morphological operations
                #     morphed_image = morph_operations(binary_image)
                #     # final_mask, rois_2 = saving_roi_from_mask(morphed_image, i-N+j)
                #     cv2.imwrite(os.path.join(foreground_path, f'fg_img_{iter+j}.png'), foreground)
                #     cv2.imwrite(os.path.join(background_path, f'bg_img_{iter+j}.png'), background)
        #             result_image2, rois_2, new_mask2 = remove_false_alarms_one_image(morphed_image, frames[iter+j], iter+j)
        #             cv2.imwrite(os.path.join(mask_alg2_path, f'mask2_{iter+j}.png'), new_mask2)
                    
        #             roiss2.append(rois_2)
        #             writer2.writerows(rois_2)
        #             masks2.append(new_mask2) 
        #             video2.write(result_image2)
        # video2.release()
                for l, (morphed_image, foreground, background) in enumerate(zip(morphed_images, foreground_images, background_images)):
                    print(type(morphed_image))
                    cv2.imwrite(os.path.join(foreground_path, f'fg_img_{iter+l}.png'), foreground)
                    cv2.imwrite(os.path.join(background_path, f'bg_img_{iter+l}.png'), background)
                    result_image2, rois_2, new_mask2 = remove_false_alarms_one_image(morphed_image, frames[iter], iter)
                    cv2.imwrite(os.path.join(mask_alg2_path, f'mask2_{iter+l}.png'), new_mask2)
                    writ_rois2 = [roi + unknown1 for roi in rois_2]
                    roiss2.append(rois_2)
                    writer2.writerows(writ_rois2)
                    masks2.append(new_mask2) 
                    video2.write(result_image2)
        video2.release()

        # # Perform element-wise multiplication of masks
        # masks = np.array(masks)
        # masks2 = np.array(masks2)
        # length = len(masks) if len(masks) >= len(masks2) else len(masks2)

        # masks_ult = np.array()

        # for i in range(length):
        #     res = masks[i] * masks2[i]
        #     res_bound = np.where(res > 0, 255, 0)
        #     masks_ult.append(res_bound)
        #     cv2.imwrite(os.path.join(mask_ult_path, f'mask_ult{i}.png'), new_mask2)
        # # Algorithm 3 (Motion Trajectory-based False Alarm Filter)
        # # Parameters
        # pipeline_length = 5
        # pipeline_size = (7, 7)
        # detection_threshold = 3
        # frame_skip_threshold = 4
        # distance_threshold = 7
end_time = time.time()

print("Time elapsed:", end_time-start_time)
