import cv2
import numpy as np
img = cv2.imread ('/Users/diana/Desktop/MMB/mot/train/046/img/000001.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
normalized = img / 255.0
eq_img = cv2.equalizeHist(gray_img)
mean, std = cv2.meanStdDev(gray_img)
std_img = (img - mean) / std
edges = cv2.Canny(gray_img, 100, 200)

resized_image = cv2.resize(img, (640, 640))
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(resized_image, -1, kernel)

# Resize image with aspect ratio maintained
height, width = img.shape[:2]
aspect_ratio = width / height
new_width = int(aspect_ratio * 640)
resized_image_ratio = cv2.resize(img, (new_width, 640), interpolation=cv2.INTER_LINEAR)


from skimage.feature import hog

# Extract HOG features
features, hog_image = hog(gray_img, visualize=True)
height, width = img.shape[:2]

def divide_into_tiles(image, tile_size):
    height, width = image.shape[:2]
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
    return tiles

# combined_image = np.hstack((gray_img, eq_img))
cv2.imshow('Image', img)
cv2.imshow('Normalized', normalized)
cv2.imshow('Grays', gray_img)
cv2.imshow(f'Equalized', eq_img)
cv2.imshow('STD_img', std_img)
cv2.imshow('Edges', edges)
cv2.imshow('Sharpened', sharpened_image)
cv2.imshow('Resize', resized_image)
cv2.imshow('hog', hog_image)
cv2.imshow('Resize aspect ratio', resized_image_ratio)
cv2.imwrite(f'/Users/diana/Desktop/MMB/mot/car/001/img/cropped_image.jpg', eq_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

