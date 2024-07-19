from algorithm1 import convert_to_binary_mask, morph_operations
from algorithm2 import construct_data_matrix, estimate_rank, fRMC

import cv2
import os

background_images_folder = '/home/diana/output_background_images'

main_path = '/home/diana/tracking/mot/'

dict_folder = {
    # 'car' : os.path.join(main_path, 'car'),
    # 'plane' : os.path.join(main_path, 'plane'),
    # 'ship' : os.path.join(main_path, 'ship'),
    'train' : os.path.join(main_path, 'train')
}

for name, name_path in dict_folder.items():
    # print(name_path)
    # print(os.listdir(name_path))
    video_folder = [os.path.join((os.path.join(name_path, path)), 'img') for path in sorted(os.listdir(name_path))]

    for video in video_folder:

    # for i in range(1, m-2):
        # im_path_list = focus_list[i:i+3]
        # print(im_path_list)
        # diff, mask = calc_difference_mask(im_path_list)
        # masks.append(mask)
        # image3 = cv2.imread(im_path_list[2], cv2.IMREAD_COLOR)
        # save_path = os.path.join(opath_im, 'sp'+ str(i) +  '.jpg')
        # mask = morph_operations(mask)
        # result_image, rois = remove_false_alarms(mask, image3, save_path)
        # roiss.append(rois)
        # resulted_imgs.append(result_image) 
        
     
        images_list = sorted(os.listdir(video))
        images_path = [os.path.join(video, image) for image in images_list]
        frames = [cv2.imread(image) for image in images_path]
        print(len(frames))
        V = construct_data_matrix(frames)
        rank = estimate_rank(V)
        B = fRMC(V, rank)
        # background_images = [B[:, i].reshape(frames[0].shape) for i in range(B.shape[1])]
        # print(background_images)
        # foreground_images = []
        # for i in len(background_images):
        #     temp_img = frames[i] - background_images[i]
        #     foreground_images.append()
        # # B = convert_to_binary_mask(B)
        # binary_images = [convert_to_binary_mask(foreground) for foreground in foreground_images]
        # morphed_images = [morph_operations(binary_image) for binary_image in binary_images]

        output_folder = os.path.join(main_path, 'output_background_images')
        os.makedirs(output_folder, exist_ok=True)
        for j in range(B.shape[1]):
            background = B[:, j].reshape(frames[0].shape)
            foreground = frames[j] - background

            # Convert to binary image
            binary_image = convert_to_binary_mask(foreground)

            # Perform morphological operations
            morphed_image = morph_operations(binary_image)
            # cv2.imwrite(os.path.join(output_folder, f'fg_img_{j}.png'), foreground)
            # cv2.imwrite(os.path.join(output_folder, f'bg_img_{j}.png'), background)
            cv2.imwrite(os.path.join(output_folder, f'img_{j}.png'), morphed_image)
            
        
        
        # # output_candidate_motion_pixels(morphed_image, output_folder)
        # for i, bg_img in enumerate(morphed_images):
        #     cv2.imwrite(os.path.join(output_folder, f'bg_img_{i}.png'), bg_img)