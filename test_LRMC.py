# import cv2
# import numpy as np
# from sklearn.decomposition import PCA
# from pyod.models.pca import PCA as RobustPCA

# def preprocess_frame(frame):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Normalize pixel values
#     normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized

# def construct_observation_matrix(frames):
#     # Stack frames as columns in the observation matrix
#     observation_matrix = np.stack(frames, axis=-1)
#     return observation_matrix

# def apply_lrmc(observation_matrix):
#     # Apply Robust PCA for low-rank matrix completion
#     print(observation_matrix.shape)
#     pca = RobustPCA(n_components=0.95, contamination=0.1)
#     pca.fit(observation_matrix)
#     low_rank_matrix = pca.components_
#     return low_rank_matrix, pca

# def detect_moving_objects(sparse_matrix):
#     # Threshold the sparse matrix to get binary mask of moving objects
#     _, binary_mask = cv2.threshold(sparse_matrix, 30, 255, cv2.THRESH_BINARY)
#     return binary_mask

# video_path = '/home/diana/mot/VISO_paper/fldout/videos/cars_test1.mp4'
# cap = cv2.VideoCapture(video_path)
# frames = []

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     processed_frame = preprocess_frame(frame)
#     frames.append(processed_frame)

# cap.release()
# # Construct observation matrix
# observation_matrix = construct_observation_matrix(frames)

# # Apply Low-Rank Matrix Completion
# low_rank_matrix, pca = apply_lrmc(observation_matrix)

# # Detect moving objects
# moving_objects_mask = detect_moving_objects(pca)

# # Display results
# for i, frame in enumerate(frames):
#     mask = moving_objects_mask[:, :, i]
#     cv2.imshow('Frame', frame)
#     cv2.imshow('Moving Objects', mask)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
# from sklearn.decomposition import PCA
from pyod.models.pca import PCA as RobustPCA


def construct_observation_matrix(frames):
    # Stack frames as columns in the observation matrix
    observation_matrix = np.stack(frames, axis=-1)
    return observation_matrix

def apply_lrmc(observation_matrix):
    # Apply Robust PCA for low-rank matrix completion
    print(observation_matrix.shape)
    pca = RobustPCA(n_components=0.95, contamination=0.1)
    pca.fit(observation_matrix)
    low_rank_matrix = pca.components_
    return low_rank_matrix, pca

def detect_moving_objects(sparse_matrix):
    # Threshold the sparse matrix to get binary mask of moving objects
    _, binary_mask = cv2.threshold(sparse_matrix, 30, 255, cv2.THRESH_BINARY)
    return binary_mask

# Path to the folder containing images
images_folder = 'mot/VISO_paper/coco/train/test2017/'

# List all image files in the folder
image_files = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')])
max_height = 0
max_width = 0
max_layers = 0
for image_file in image_files:
    frame = cv2.imread(image_file)
    img_height, img_width, layers = frame.shape
    if img_height > max_height:
        max_height = img_height
    if img_width > max_width:
        max_width = img_width
    if layers > max_layers:
        max_layers = layers
print(max_height, max_width, max_layers)

frames = [cv2.resize(cv2.imread(image_file), (max_height, max_width)) for image_file in image_files]

# Construct observation matrix
observation_matrix = construct_observation_matrix(frames)

# Apply Low-Rank Matrix Completion
low_rank_matrix, pca = apply_lrmc(observation_matrix)

# Detect moving objects
moving_objects_mask = detect_moving_objects(pca)
print("Help")
# Display results
for i, image_file in enumerate(image_files):
    frame = cv2.imread(image_file)
    mask = moving_objects_mask[:, :, i]
    cv2.imshow('Frame', frame)
    cv2.imshow('Moving Objects', mask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
