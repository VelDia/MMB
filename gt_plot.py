from plotting_detections import read_csv_file_alg2
import os
import numpy as np
import cv2

gt_path_low = "voc_dataset/low"
gt_path_low_img = "/home/diana/MMB/voc_dataset/train/images/000001.jpg"
image = cv2.imread(gt_path_low_img)
width, height, _ = image.shape

gt_path_low_txt = "/home/diana/MMB/voc_dataset/train/labels/000001.txt"

gt = read_csv_file_alg2(gt_path_low_txt)
print(gt[0][0])
gt = np.array(gt[0][0], dtype=list)
gt_xmin, gt_ymin, gt_xmax, gt_ymax = map(int, gt)
cv2.rectangle(image, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0),3)

cv2.imshow('ROI', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
# # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

# # area = stats[label, cv2.CC_STAT_AREA]
# # x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
# # aspect_ratio = width / height if height != 0 else 0  # Calculate aspect ratio
# # print(preds_alg2[200])
# for gt_box, detection, yolo_det in zip(ground_truths[200], preds[200], yolo_preds[200]):
#     gt_xmin, gt_ymin, gt_xmax, gt_ymax = map(int, gt_box)
#     xmin, ymin, xmax, ymax = map(int, detection)
#     # xmin2, ymin2, xmax2, ymax2 = map(int, det_alg2)
#     xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = map(int, yolo_det)

#     cv2.rectangle(image, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0),3)
#     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
#     # cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), (0, 255, 255), 4)
#     cv2.rectangle(image, (xmin_yolo, ymin_yolo), (xmax_yolo, ymax_yolo), (0, 0, 255), 2)
# legend_entries = [
#     ("Ground truth", (0, 255, 0)),
#     ("AMFD", (255, 0, 0)),
#     # ("Another Box", (0, 255, 255)),
#     ("YOLO", (0, 0, 255))
# ]

# # Positioning for the legend
# legend_x = 10
# legend_y = 30
# line_height = 30

# # Add text for each legend entry
# for i, (label, color) in enumerate(legend_entries):
#     # Draw a small rectangle for the color
#     cv2.rectangle(image, (legend_x, legend_y + i * line_height - 20), (legend_x + 20, legend_y + i * line_height - 10), color, -1)
#     # Put text next to the rectangle
#     cv2.putText(image, label, (legend_x + 30, legend_y + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)

# cv2.imshow('ROI', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()