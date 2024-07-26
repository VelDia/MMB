import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_difference_mask (im_path_list):
    # read images
    image1 = cv2.imread(im_path_list[0], cv2.IMREAD_COLOR)
    image2 = cv2.imread(im_path_list[1], cv2.IMREAD_COLOR)
    image3 = cv2.imread(im_path_list[2], cv2.IMREAD_COLOR)
    image4 = cv2.imread(im_path_list[3], cv2.IMREAD_COLOR)
    image5 = cv2.imread(im_path_list[4], cv2.IMREAD_COLOR)

    # Calculate differencing images
    D_t1 = np.abs(image3.astype(np.int32) - image1.astype(np.int32))
    D_t2 = np.abs(image5.astype(np.int32) - image1.astype(np.int32))
    D_t3 = np.abs(image5.astype(np.int32) - image3.astype(np.int32))
    D_t4 = np.abs(image4.astype(np.int32) - image2.astype(np.int32))
    D_t5 = np.abs(image4.astype(np.int32) - image3.astype(np.int32))

    # Calculate the accumulative response image 
    # acc_resp_im = (D_t1 + D_t2 + D_t3)/3
    acc_resp_im = (D_t1 + D_t2 + D_t3 +D_t4 +D_t5)/5

    # Calculate threshold T to extract targets
    mu = np.mean(acc_resp_im) # the mean of the differencing image acc_resp_im
    std_I = np.std(acc_resp_im) # the standard deviation of differencing image acc_resp_im
    k = 6 # coefficient / hyperparameter (empirical)
    threshold = mu + k * std_I
    print(threshold)

    # Convert the acc_resp_im to a binary image
    diff_gray = cv2.cvtColor(acc_resp_im.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(diff_gray, dtype=np.uint8)
    mask[diff_gray >= threshold] = 255  

    return acc_resp_im, mask


def plot_mask (im_path_list):

    image1 = cv2.imread(im_path_list[0], cv2.IMREAD_COLOR)
    image2 = cv2.imread(im_path_list[2], cv2.IMREAD_COLOR)
    diff, mask = calc_difference_mask(im_path_list)

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

def morph_operations(mask, orig_image, save_path):
    min_area = 4
    max_area = 324
    min_aspect_ratio = 0.25
    max_aspect_ratio = 6.0
    
    output_image = orig_image.copy()

    rois = []
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        aspect_ratio = width / height if height != 0 else 0  # Calculate aspect ratio

        # Check if component meets area and aspect ratio criteria
        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            # Process or store the component (e.g., draw bounding box, compute other properties)
            print(f"Label: {label}, Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")

            # Example: Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            # Extract ROI from original image based on bounding box coordinates
            roi = orig_image[y:y+height, x:x+width]

            # Append ROI to list
            rois.append(roi)

    if save_path is not None:
        cv2.imwrite(save_path, orig_image)
        
    
    return output_image, rois
