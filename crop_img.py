import cv2
import numpy as np

def crop_image_into_parts(image, crop_width, crop_height):
    # Get image dimensions
    height, width, _ = image.shape
    
    # List to store cropped images
    cropped_images = []
    
    # Loop to crop the image into equal parts
    for y in range(0, height, crop_height):
        for x in range(0, width, crop_width):
            # Define the end coordinates
            end_x = min(x + crop_width, width)
            end_y = min(y + crop_height, height)
            
            # Crop the image
            crop = image[y:end_y, x:end_x]
            cropped_images.append(crop)
    
    return cropped_images

# Read the image
image = cv2.imread('path/to/your/image.jpg')

# Define crop size
crop_width = 100  # Width of each crop
crop_height = 100 # Height of each crop

# Get the cropped parts
cropped_images = crop_image_into_parts(image, crop_width, crop_height)

# Display or save each cropped part
for i, crop in enumerate(cropped_images):
    cv2.imshow(f'Cropped Image {i+1}', crop)
    cv2.imwrite(f'path/to/save/cropped_image_{i+1}.jpg', crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
