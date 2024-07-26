import cv2
import numpy as np
import matplotlib.pyplot as plt

im1_path = 'mot/VISO_paper/coco/car/test2017/000001.jpg'
im2_path = 'mot/VISO_paper/coco/car/test2017/000003.jpg'

# Load images (make sure they are of the same size)
image1 = cv2.imread(im1_path, cv2.IMREAD_COLOR)
image2 = cv2.imread(im2_path, cv2.IMREAD_COLOR)

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
plt.title('Image 1')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Image 2')

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



