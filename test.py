from algorithm1 import calc_difference_mask, remove_false_alarms_one_image, convert_to_binary_mask, morph_operations
from algorithm2 import construct_data_matrix, estimate_rank, fRMC, calc_num_observ_matrix
import math
import cv2
import os
import csv

# list of paths
opath_im = '/Users/diana/Desktop/tracking/MMB/output_rois'
save_preds_p1 = 'pred_alg1.txt'
save_preds_p1 = os.path.join(opath_im, save_preds_p1)
save_preds_p2 = 'pred_alg2.txt'
save_preds_p2 = os.path.join(opath_im, save_preds_p2)
video_path = os.path.join(opath_im, 'video')
foreground_path = os.path.join(opath_im, 'foreground')
background_path = os.path.join(opath_im, 'background')
mask_alg1_path = os.path.join(opath_im, 'masks_alg1')
os.makedirs(mask_alg1_path, exist_ok=True)
os.makedirs(opath_im, exist_ok=True)
os.makedirs(video_path, exist_ok=True)
video_path = os.path.join(video_path, 'video.mp4')

# getting frames
dataset_path = '/Users/diana/Desktop/tracking/MMB/mot/car/001/img'
images_list = sorted(os.listdir(dataset_path))
images_path = [os.path.join(dataset_path, image) for image in images_list if image != '.DS_Store']
frames = [cv2.imread(image) for image in images_path]
m = len(frames)
print(m)

masks = []
roiss = []
resulted_imgs = []
width, height, _ = frames[0].shape

# # save the video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

# with open(save_preds_p1, 'w', newline='') as file:
#     writer = csv.writer(file)

#     # Algorithm #1
#     for i in range(1, m-2):
#         im_path_list = images_path[i:i+3]
#         # print(im_path_list)
#         image = cv2.imread(im_path_list[0], cv2.IMREAD_COLOR)

#         diff, mask = calc_difference_mask(im_path_list)
#         masks.append(mask)

#         mask = morph_operations(mask)
#         result_image, rois, new_mask = remove_false_alarms_one_image(mask, image, i)
#         cv2.imwrite(os.path.join(mask_alg1_path, f'mask_{i}.png'), new_mask)
#         roiss.append(rois)
#         writer.writerows(rois)
#         resulted_imgs.append(result_image) 
#         video.write(result_image)
#     video.release()

# Algorithm 2
with open(save_preds_p2, 'w', newline='') as file:
    writer = csv.writer(file)

    # Calculate the number of observation matrices 
    N = calc_num_observ_matrix(frames)
    print(N)
    # print(len(frames[N-N:N]))
    foreground_images = []
    for i in range(N, m, N):
        V = construct_data_matrix(frames[i-N:i])
        rank = estimate_rank(V)
        B = fRMC(V, rank)
        background_images = [B[:, i].reshape(frames[0].shape) for i in range(B.shape[1])]

        for k in range(len(background_images)):
            temp_img = frames[k] - background_images[k]
            foreground_images[i-N+k] = temp_img
        
        # B = convert_to_binary_mask(B)
        binary_images = [convert_to_binary_mask(foreground) for foreground in foreground_images]
        morphed_images = [morph_operations(binary_image) for binary_image in binary_images]

        for j in range(B.shape[1]):
            background = B[:, j].reshape(frames[0].shape)
            foreground = frames[j] - background

            # Convert to binary image
            binary_image = convert_to_binary_mask(foreground)

            # Perform morphological operations
            morphed_image = morph_operations(binary_image)
            # cv2.imwrite(os.path.join(output_folder, f'fg_img_{j}.png'), foreground)
            # cv2.imwrite(os.path.join(output_folder, f'bg_img_{j}.png'), background)
            cv2.imwrite(os.path.join(background_path, f'bg_{j}.png'), morphed_image)
            writer.writerows(rois)


        # # output_candidate_motion_pixels(morphed_image, output_folder)
        # for i, bg_img in enumerate(morphed_images):
        #     cv2.imwrite(os.path.join(output_folder, f'bg_img_{i}.png'), bg_img)