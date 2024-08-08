# def calculate_iou(box1, box2):
#     """
#     Function to calculate IoU (Intersection over Union) between two bounding boxes.
#     Each bounding box is expected to be in the format:
#     (xmin, ymin, xmax, ymax)
#     """
#     x1_topleft, y1_topleft, x1_bottomright, y1_bottomright = box1
#     x2_topleft, y2_topleft, x2_bottomright, y2_bottomright = box2

#     # Calculate intersection coordinates
#     x_left = max(x1_topleft, x2_topleft)
#     y_top = max(y1_topleft, y2_topleft)
#     x_right = min(x1_bottomright, x2_bottomright)
#     y_bottom = min(y1_bottomright, y2_bottomright)

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0  # No overlap

#     # Calculate area of intersection rectangle
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # Calculate area of both bounding boxes
#     box1_area = (x1_bottomright - x1_topleft) * (y1_bottomright - y1_topleft)
#     box2_area = (x2_bottomright - x2_topleft) * (y2_bottomright - y2_topleft)

#     # Calculate IoU
#     iou = intersection_area / float(box1_area + box2_area - intersection_area)
#     return iou

# def is_true_positive(pred_box, gt_box, threshold=0.5):
#     """
#     Function to determine if a predicted bounding box is a True Positive based on IoU threshold.
#     pred_box and gt_box should be in format (xmin, ymin, xmax, ymax).
#     """
#     iou = calculate_iou(pred_box, gt_box)
#     if iou >= threshold:
#         return True  # True Positive
#     else:
#         return False  # Not a True Positive

import csv
import numpy as np
def calculate_iou(box1, box2):
    x1_topleft, y1_topleft, x1_bottomright, y1_bottomright = box1
    x2_topleft, y2_topleft, x2_bottomright, y2_bottomright = box2

    x_left = max(x1_topleft, x2_topleft)
    y_top = max(y1_topleft, y2_topleft)
    x_right = min(x1_bottomright, x2_bottomright)
    y_bottom = min(y1_bottomright, y2_bottomright)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_bottomright - x1_topleft) * (y1_bottomright - y1_topleft)
    box2_area = (x2_bottomright - x2_topleft) * (y2_bottomright - y2_topleft)
    
    # print(intersection_area)
    if intersection_area > 0:
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
    else:
        iou = 0
    return iou

def is_true_positive(pred_box, gt_box, threshold=0.1):
    iou = calculate_iou(pred_box, gt_box)
    return iou >= threshold

def calculate_true_positives(predictions, ground_truths, threshold=0.1):
    """
    Function to calculate the number of True Positives (TP) for each algorithm's predictions.
    predictions: List of lists of bounding boxes from the algorithm.
    ground_truths: List of ground truth bounding boxes.
    threshold: IoU threshold for considering a prediction as True Positive.
    """
    true_positives = 0
    used_gt_boxes = [False] * len(ground_truths)  # Track which ground truth boxes have been used

    for pred_box in predictions:
        # matched = False
        for i, gt_box in enumerate(ground_truths):
            if not used_gt_boxes[i] and is_true_positive(pred_box, gt_box, threshold):
                true_positives += 1
                used_gt_boxes[i] = True  # Mark this ground truth box as used
                # matched = True
                break  # Move to the next prediction

    return true_positives

def calculate_tp_fp_fn(predictions, ground_truths, threshold=0.1):
    true_positives = calculate_true_positives(predictions, ground_truths, threshold)
    total_ground_truths = len(ground_truths)
    total_predictions = len(predictions)
    false_negatives = total_ground_truths - true_positives
    false_positives = total_predictions - true_positives

    return true_positives, false_positives, false_negatives

def calculate_metrics(all_true_boxes, all_predicted_boxes, iou_threshold=0.1):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Iterate through all images
    for true_boxes, predicted_boxes in zip(all_true_boxes, all_predicted_boxes):
        tp, fp, fn = calculate_tp_fp_fn(true_boxes, predicted_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculate aggregated metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def read_csv_file_gt(file_path):
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

            j, _, xmin, ymin, height, width, _, _, _, _ = map(float, row)
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

            j, _, xmin, ymin, height, width = map(float, row)
            if i != j:
                bounding_boxes.append(bb_img)
                bb_img=[]
                i = j
            xmax = xmin + height
            ymax = ymin + width
            
            bb_img.append([xmin, ymin, xmax, ymax])
            
    return bounding_boxes

gt_path = 'mot/car/001/gt/gt.txt'
# pred_path = 'output_rois_new/car/1/pred_alg1.txt'
pred_path = 'output_rois_orig/pred_alg2.txt'

ground_truths = read_csv_file_gt(gt_path)
ground_truths = np.array(ground_truths, dtype=object)

preds = read_csv_file(pred_path)
preds = np.array(preds, dtype=object)
# Possible iteration loops to access data

# Loop Num 1
for image in ground_truths:
    for coordinates in image:
        break
        print(coordinates)

# Loop Num 2
for i in range(len(preds)):
    # 'i' is the id of the frame
    for j in range(len(ground_truths[i])): 
        # 'j' is the id of the individual bounding boxes 
        
        for k in range(len(ground_truths[i][j])):
            break
            # print(ground_truths[i][j][k])
eval = calculate_metrics(preds, ground_truths)
print(eval)


# iou_test = calculate_iou(ground_truths[i][j], ground_truths[i][j])
# is_tp_test = is_true_positive(ground_truths[i][j], ground_truths[i][j])
# print(is_tp_test)
# tps = calculate_true_positives(preds[i], ground_truths[i])
# # print(tps, '/', len(preds[i]), '/', len(ground_truths[i]))
# eval = calculate_metrics(preds, ground_truths)
# print(eval)