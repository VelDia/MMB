import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_binary_mask(img):

    # Calculate threshold T to extract targets
    mu = np.mean(img) # the mean of the differencing image acc_resp_im
    std_I = np.std(img) # the standard deviation of differencing image acc_resp_im
    k = 4 # coefficient / hyperparameter (empirical)
    threshold = mu + k * std_I
    # print(threshold)

    # Convert the image to a binary image
    diff_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(diff_gray, dtype=np.uint8)
    mask[diff_gray >= threshold] = 255  

    return mask

def calc_difference_mask (im_path_list):
    # read images
    images = []
    for img in im_path_list:
        images.append(cv2.imread(img, cv2.IMREAD_COLOR))
    num = len(images)
    # print(round(num/2)) # Check if the middle image has the intended position
    # Calculate differencing images
    D_t1 = np.abs(images[round(num/2)].astype(np.int32) - images[0].astype(np.int32))
    D_t2 = np.abs(images[-1].astype(np.int32) - images[0].astype(np.int32))
    D_t3 = np.abs(images[-1].astype(np.int32) - images[round(num/2)].astype(np.int32))

    # Calculate the accumulative response image 
    acc_resp_im = (D_t1 + D_t2 + D_t3)/3
    
    # Get the mask of moving objects
    mask = convert_to_binary_mask(acc_resp_im)
    mask = morph_operations(mask)
    return acc_resp_im, mask

def morph_operations(image):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def plot_mask(im_path_list):

    image1 = cv2.imread(im_path_list[0], cv2.IMREAD_COLOR)
    image2 = cv2.imread(im_path_list[-1], cv2.IMREAD_COLOR)
    diff, mask = calc_difference_mask(im_path_list)
    mask = morph_operations(mask)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image First')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('Image Last')

    plt.subplot(2, 2, 3)
    plt.imshow(diff.astype(np.uint8))
    plt.title('Difference Image')

    plt.subplot(2,2,4)
    plt.imshow(mask, cmap='gray')
    plt.title('Difference Mask')
    plt.colorbar()

    plt.show()

def remove_false_alarms_one_image(mask, orig_image, num_im):
    
    # Original parameters

    # min_area = 5
    # max_area = 80
    # min_aspect_ratio = 1.0
    # max_aspect_ratio = 6.0

    # Custom parameters
    min_area = 4
    max_area = 324
    min_aspect_ratio = 0.25
    max_aspect_ratio = 6.0

    # Create a copy of the original image to draw bounding boxes
    output_image = orig_image.copy()

    # List to store extracted ROIs
    rois = []
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        aspect_ratio = width / height if height != 0 else 0  # Calculate aspect ratio

        # Check if component meets area and aspect ratio criteria
        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:

            # Process or store the component (e.g., draw bounding box, compute other properties)
            # print(f"Label: {label}, Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")

            # Example: Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0))

            # cv2.imshow('ROI', output_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Draw the component on the new mask
            new_mask[labels == label] = 255

            # Extract ROI from original image based on bounding box coordinates
            roi = orig_image[x:x+width+1, y:y+height+1]

            # Append ROI to list
            rois.append([int(num_im), int(label), int(x), int(y), int(width), int(height)]) #append coordinates as they appear in (ground truth) gt.txt
            # print([int(num_im), int(label), int(x), int(y), int(width), int(height)])

    # cv2.destroyAllWindows()
    return output_image, rois, new_mask

def load_images_from_folder(images_folder):

    # List all image files in the folder
    image_files = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg')])
    frames = [cv2.imread(image_file) for image_file in image_files]
    return frames
