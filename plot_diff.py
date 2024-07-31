import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from algorithm1 import calc_difference_mask, morph_operations

# Load images (make sure they are of the same size)
folder_path = '/Users/diana/Desktop/MMB/mot/car/025/img'
images_list = sorted(os.listdir(folder_path))
images_path = [os.path.join(folder_path, image) for image in images_list if image != '.DS_Store']
frames = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images_path]
image1 = frames[0]
image2 = frames[-1]

# # Compute absolute difference
# diff = np.abs(image1.astype(np.int32) - image2.astype(np.int32))

# # Optionally, calculate a total difference score
# total_diff = np.sum(diff)

# # Display or save the difference image
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image1, cmap='gray')
# plt.title('Image 1')

# plt.subplot(1, 2, 2)
# plt.imshow(diff, cmap='gray')
# plt.title('Difference Image')

# plt.tight_layout()
# plt.show()
image1 = cv2.imread(images_path[0], cv2.IMREAD_COLOR)
image2 = cv2.imread(images_path[-1], cv2.IMREAD_COLOR)

diff, mask = calc_difference_mask(images_path)
mask = morph_operations(mask)

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Image First')

plt.subplot(1, 2, 2)
plt.imshow(diff.astype(np.uint8))
plt.title('Difference Image')

plt.show()
