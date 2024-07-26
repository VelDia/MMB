import cv2
import numpy as np
import matplotlib.pyplot as plt

im001_path = 'mot/VISO_paper/coco/car/test2017/000001.jpg'
im325_path = 'mot/VISO_paper/coco/car/test2017/000325.jpg'
im010_path = 'mot/VISO_paper/coco/car/test2017/000010.jpg'
# for im in 

# Load images (make sure they are of the same size)
image1 = cv2.imread(im001_path, cv2.IMREAD_COLOR)
image2 = cv2.imread(im325_path, cv2.IMREAD_COLOR)

# Compute absolute difference
diff = np.abs(image1.astype(np.int32) - image2.astype(np.int32))

# Optionally, calculate a total difference score
total_diff = np.sum(diff)

# Convert difference to grayscale for mask creation
diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

# Define a threshold for the mask
threshold = 50  # Adjust this threshold as needed

# Create a mask where differences are above the threshold
mask = np.zeros_like(diff_gray, dtype=np.uint8)
mask[diff_gray > threshold] = 255  # White color for differences above threshold

colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
colored_mask[mask > 0] = [0, 0, 255] 

# Display or save the difference image
# plt.figure(figsize=(12, 10))
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

# plt.subplot(2,2,4)
# plt.imshow(mask, cmap='gray')
# plt.title('Difference Mask')
# plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
plt.title('Colored Mask')
plt.colorbar()

plt.tight_layout()
plt.show()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the folder path
folder_path = 'mot/VISO_paper/coco/car/test2017/'

# Step 2: Read images from the folder
image_files = os.listdir(folder_path)
image_files.sort()  # Sort filenames to ensure images are in the correct order

# Step 3: Load images into a list
images = []
for filename in image_files:

    if filename.endswith('.jpg'):  # Adjust extensions as needed
        filename_path = os.path.join(folder_path, filename)
        if filename_path == im010_path:
            break
        image = cv2.imread(filename_path, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
        else:
            print(f"Error loading {filename}")

if len(images) > 1:
    accumulator = np.zeros_like(images[0], dtype=np.float64)
    num_masks = 0

    for i in range(len(images) - 1):
        # Compute absolute difference in RGB space
        # diff = np.abs(images[i].astype(np.int32) - images[i+1].astype(np.int32))

        # # Convert difference to grayscale for mask creation
        # diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # # Define a threshold for the grayscale mask
        threshold = 230  # Adjust this threshold as needed

        # # Create a binary mask where differences are above the threshold
        # mask = np.zeros_like(diff_gray, dtype=np.uint8)
        # mask[diff_gray > threshold] = 255  # White color for differences above threshold
        #  # Accumulate mask into accumulator
        # accumulator += mask.astype(np.float64)
        # num_masks += 1
        diff = np.abs(images[i].astype(np.int32) - images[i+1].astype(np.int32))

        # Accumulate differences across RGB channels
        accumulator += diff.astype(np.float64)

        num_masks += 1
    # Step 5: Average masks
    if num_masks > 0:
        average_mask = (accumulator / num_masks).astype(np.uint8)
        
        # Plot the average mask
        plt.figure(figsize=(6, 6))
        # plt.subplot(1,2,1)
        # plt.imshow(average_mask, cmap='gray')
        # plt.title('Average Difference Mask')
        # plt.axis('off')

        average_mask[accumulator>threshold] =255
        # plt.subplot(1,2,2)
        plt.imshow(average_mask, cmap='gray')
        plt.title('Average Difference Mask')
        plt.show()

        # Threshold the image to binary
        _, binary_image = cv2.threshold(average_mask, 1, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        # Filter out small components
        min_size = 5  # Minimum size of components to keep
        filtered_image = np.zeros_like(average_mask)


        # Display the original and filtered images
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(average_mask, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Filtered Image")
        plt.imshow(filtered_image, cmap='gray')

        plt.show()

else:
    print("Error: Insufficient images found in the folder.")
