import cv2
import os
import numpy as np
from part3 import calculate_true_positives, calculate_metrics, read_csv_file
from algorithm2 import saving_roi_from_mask

# list of paths
opath_im = 'output_rois'
# opath_im = '/Users/diana/Desktop/tracking/MMB/output_rois_orig'
# opath_im = '/home/diana/MMB/output_rois'
# opath_im = '/home/diana/MMB/output_rois_limited'

# Perform element-wise multiplication of masks
mask_alg1_path = os.path.join(opath_im, 'masks_alg1')
mask_alg2_path = os.path.join(opath_im, 'masks_alg2')
mask_ult_path = os.path.join(opath_im, 'masks_ult')
os.makedirs(mask_ult_path, exist_ok=True)

masks = [cv2.imread(os.path.join(mask_alg1_path, img_path)) for img_path in os.listdir(mask_alg1_path)]
masks2 = [cv2.imread(os.path.join(mask_alg2_path, img_path)) for img_path in os.listdir(mask_alg2_path)]

np_masks = np.array(masks)
np_masks2 = np.array(masks2)
length = len(masks) if len(masks) <= len(masks2) else len(masks2)
np_masks_ult = np.array([])

for i in range(length-1):
    res = np_masks[i] * np_masks2[i]
    res_bound = np.where(res > 0, 255, 0)
    masks_ult = np.append(np_masks_ult, res_bound)
    cv2.imwrite(os.path.join(mask_ult_path, f'mask_ult{i}.png'), res_bound)

roi1 = []
for mask in masks:
    roi1.append(saving_roi_from_mask(mask))

roi2 = []
for mask in masks2:
    roi2.append(saving_roi_from_mask(mask))

roi3 = []
for mask in masks_ult:
    roi3.append(saving_roi_from_mask(mask))

# Example data
gt_path = 'mot/car/001/gt/gt.txt'

# Read data from files
# algorithm1_predictions = read_csv_file('algorithm1_predictions.txt')
# algorithm2_predictions = read_csv_file('algorithm2_predictions.txt')
ground_truths = read_csv_file(gt_path)
# print(ground_truths)
algorithm1_predictions = roi1
algorithm2_predictions = roi2

print(len(ground_truths))
print(len(roi1))
print(len(roi2))
# Calculate TP for each algorithm
# tp_algorithm1 = calculate_true_positives(algorithm1_predictions, ground_truths, threshold=0.5)
# tp_algorithm2 = calculate_true_positives(algorithm2_predictions, ground_truths, threshold=0.5)

# print(f"True Positives for Algorithm 1: {tp_algorithm1}")
# print(f"True Positives for Algorithm 2: {tp_algorithm2}")

# precision1, recall1, f1_score1 = calculate_metrics(roi1, ground_truths, 0.01)
# precision2, recall2, f1_score2 = calculate_metrics(roi2, ground_truths, 0.01)
# print("Alg 1: ", precision1, recall1, f1_score1)
# print("Alg 2: ", precision2, recall2, f1_score2)
# # Algorithm 3 (Motion Trajectory-based False Alarm Filter)
# # Parameters
# pipeline_length = 5
# pipeline_size = (7, 7)
# detection_threshold = 3
# frame_skip_threshold = 4
# distance_threshold = 7

# # for i in range(0, len(frames), )
# # for i in range(pipeline_length):
# #     current_frame = frames[i]
# rois1 = []
# # threshold = 0.5  # IoU threshold
# # result = is_true_positive(predicted_box, ground_truth_box, threshold)
# # print("Is True Positive:", result)