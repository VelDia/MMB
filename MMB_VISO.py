cars_test_path = 'mot/VISO_paper/coco/car/test2017'
planes_test_path = 'mot/VISO_paper/coco/plane/test2017'
ships_test_path = 'mot/VISO_paper/coco/ship/test2017'
trains_test_path = 'mot/VISO_paper/coco/train/test2017'

## AMFD
import cv2
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images (make sure they are of the same size)
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Compute absolute difference
diff = np.abs(image1.astype(np.int32) - image2.astype(np.int32))

# Optionally, calculate a total difference score
total_diff = np.sum(diff)

# Display or save the difference image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(diff, cmap='gray')
plt.title('Difference Image')

plt.tight_layout()
plt.show()

