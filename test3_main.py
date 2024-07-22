import cv2
import os
import numpy as np
from part3 import is_true_positive 
from algorithm2 import saving_roi_from_mask

# list of paths
opath_im = '/home/diana/MMB/output_rois'

# Perform element-wise multiplication of masks
mask_alg1_path = os.path.join(opath_im, 'masks_alg1')
mask_alg2_path = os.path.join(opath_im, 'masks_alg2')
mask_ult_path = os.path.join(opath_im, 'masks_ult')
os.makedirs(mask_ult_path, exist_ok=True)

masks = [cv2.imread(os.path.join(mask_alg1_path, img_path)) for img_path in os.listdir(mask_alg1_path)]
masks2 = [cv2.imread(os.path.join(mask_alg2_path, img_path)) for img_path in os.listdir(mask_alg2_path)]

masks = np.array(masks)
masks2 = np.array(masks2)
length = len(masks) if len(masks) <= len(masks2) else len(masks2)
masks_ult = np.array([])

for i in range(length-1):
    res = masks[i] * masks2[i]
    res_bound = np.where(res > 0, 255, 0)
    masks_ult = np.append(masks_ult, res_bound)
    cv2.imwrite(os.path.join(mask_ult_path, f'mask_ult{i}.png'), res_bound)

# Algorithm 3 (Motion Trajectory-based False Alarm Filter)
# Parameters
pipeline_length = 5
pipeline_size = (7, 7)
detection_threshold = 3
frame_skip_threshold = 4
distance_threshold = 7

# for i in range(0, len(frames), )
# for i in range(pipeline_length):
#     current_frame = frames[i]
rois1 = []
# threshold = 0.5  # IoU threshold
# result = is_true_positive(predicted_box, ground_truth_box, threshold)
# print("Is True Positive:", result)