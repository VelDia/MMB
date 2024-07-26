import os
import cv2
from lib_paper import plot_mask, calc_difference_mask, remove_false_alarms, morph_operations

cars_test_path = 'mot/VISO_paper/coco/car/test2017/'
planes_test_path = 'mot/VISO_paper/coco/plane/test2017'
ships_test_path = 'mot/VISO_paper/coco/ship/test2017'
trains_test_path = 'mot/VISO_paper/coco/train/test2017'

cars_images_list = sorted(os.listdir(cars_test_path))
cars_images_fullpath = [os.path.join(cars_test_path, image_file) for image_file in cars_images_list]
cars_images_fullpath.sort()

focus_list = cars_images_fullpath

# testing
im3_list = cars_images_fullpath[0:3]
print(im3_list)
plot_mask(im3_list)
diff, mask = calc_difference_mask(im3_list)
image3 = cv2.imread(im3_list[2], cv2.IMREAD_COLOR)

# image3 = cv2.imread(im3_list[2], cv2.IMREAD_COLOR)
save_path = 'output_image_with_boxes.png'

# Perform morphological operations and save the result
result_image, rois = remove_false_alarms(mask, image3, save_path)

# Display or further process the result_image and rois as needed
cv2.imshow('Result with Bounding Boxes', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display or process each ROI
for i, roi in enumerate(rois):
    cv2.imshow(f'ROI {i+1}', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# end testing


m = len(cars_images_list)
roiss = []
resulted_imgs = []
masks = []
opath_im = '/home/diana/mot/VISO_paper/fldout/imgs_test/'


'''
I have the images with bounding boxes drawn by cv2.rectangle(). How do I save them in the file with bounding boxes? 
'''

for i in range(1, m-2):
    im_path_list = focus_list[i:i+3]
    print(im_path_list)
    diff, mask = calc_difference_mask(im_path_list)
    masks.append(mask)
    image3 = cv2.imread(im_path_list[2], cv2.IMREAD_COLOR)
    save_path = os.path.join(opath_im, 'sp'+ str(i) +  '.jpg')
    mask = morph_operations(mask)
    result_image, rois = remove_false_alarms(mask, image3, save_path)
    roiss.append(rois)
    resulted_imgs.append(result_image)  
                     



    frame = result_image
    height, width, layers = frame.shape

# Initialize the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

# Write each image to the video
for image in image_files:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release the VideoWriter
video.release()

# Clean up
cv2.destroyAllWindows()