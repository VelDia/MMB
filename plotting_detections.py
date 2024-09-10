import csv
import numpy as np
import cv2
import os

def read_csv_file_yolo(file_path, img_width, img_height):
    """
    Reads a CSV file and returns a list of bounding boxes.
    Each line is expected to be in the format: xmin,ymin,xmax,ymax
    """
    bounding_boxes = []
    # for file_path in files:
    with open(file_path, 'r') as file:
        bb_img =[]
        for line in file:
            # Strip leading/trailing whitespace and split the line by spaces
            parts = line.strip().split()
            # Convert each part to a float and store it in a list
            float_list = [float(part) for part in parts]

            _, xcentr, ycentr, height, width = float_list
            
            xmax = xcentr + height/2
            xmin = xcentr - height/2
            ymax = ycentr + width/2
            ymin = ycentr - width/2
            
            bb_img.append([xmin*img_width, ymin*img_height, xmax*img_width, ymax*img_height])
    bounding_boxes.append(bb_img)

    return bounding_boxes

def read_csv_file_alg2(file_path):
    """
    Reads a CSV file and returns a list of bounding boxes.
    Each line is expected to be in the format: xmin,ymin,xmax,ymax
    """
    bounding_boxes = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        bb_img =[]
        for row in reader:

            j, _, xmin, ymin, height, width = map(int, row)
            if i != j:
                bounding_boxes.append(bb_img)
                bb_img=[]
                i = j
            xmax = xmin + height
            ymax = ymin + width
            
            bb_img.append([xmin, ymin, xmax, ymax])
            
    return bounding_boxes

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a list of bounding boxes.
    Each line is expected to be in the format: xmin,ymin,xmax,ymax
    """
    bounding_boxes = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # next(reader)  # Skip the header row
        i = 0
        bb_img =[]
        for row in reader:

            j, _, xmin, ymin, height, width, _, _, _, _ = map(int, row)
            if i != j:
                bounding_boxes.append(bb_img)
                bb_img=[]
                i = j
            xmax = xmin + height
            ymax = ymin + width
            
            bb_img.append([xmin, ymin, xmax, ymax])
            
    return bounding_boxes
'''
gt_path = 'mot/car/001/gt/gt.txt'
ground_truths = read_csv_file(gt_path)
ground_truths = np.array(ground_truths, dtype=object)
path_to_img = 'mot/car/001/img/000201.jpg'
image = cv2.imread(path_to_img)

pred_path = 'output_rois_new/car/1/pred_alg1.txt'
preds = read_csv_file(pred_path)
preds = np.array(preds, dtype=object)

# # pred_path_alg2 = 'output_rois_any/car/mot/car/001/img/pred_alg2.txt'
# pred_path_alg2 = 'output_rois/car/1/pred_alg2.txt'
# # preds_alg2 = read_csv_file_alg2(pred_path_alg2)
# preds_alg2 = read_csv_file(pred_path_alg2)
# preds_alg2 = np.array(preds_alg2, dtype=object)

width, height, _ = image.shape
# path_yolo_pred = '/Users/diana/Desktop/MMB/mot/car/001/img/output_yolov5_orig/000201.txt'

path = 'mot/car/001/img/output_yolov5_orig'
txt_list = sorted(os.listdir(path))
txt_path = [os.path.join(path, file) for file in txt_list if file.endswith('.txt')]
yolo_preds = read_csv_file_yolo(txt_path, width, height)
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

# area = stats[label, cv2.CC_STAT_AREA]
# x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
# aspect_ratio = width / height if height != 0 else 0  # Calculate aspect ratio
# print(preds_alg2[200])
for gt_box, detection, yolo_det in zip(ground_truths[200], preds[200], yolo_preds[200]):
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = map(int, gt_box)
    xmin, ymin, xmax, ymax = map(int, detection)
    # xmin2, ymin2, xmax2, ymax2 = map(int, det_alg2)
    xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = map(int, yolo_det)

    cv2.rectangle(image, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0),3)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    # cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), (0, 255, 255), 4)
    cv2.rectangle(image, (xmin_yolo, ymin_yolo), (xmax_yolo, ymax_yolo), (0, 0, 255), 2)
legend_entries = [
    ("Ground truth", (0, 255, 0)),
    ("AMFD", (255, 0, 0)),
    # ("Another Box", (0, 255, 255)),
    ("YOLO", (0, 0, 255))
]

# Positioning for the legend
legend_x = 10
legend_y = 30
line_height = 30

# Add text for each legend entry
for i, (label, color) in enumerate(legend_entries):
    # Draw a small rectangle for the color
    cv2.rectangle(image, (legend_x, legend_y + i * line_height - 20), (legend_x + 20, legend_y + i * line_height - 10), color, -1)
    # Put text next to the rectangle
    cv2.putText(image, label, (legend_x + 30, legend_y + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)

cv2.imshow('ROI', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
