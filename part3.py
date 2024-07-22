def calculate_iou(box1, box2):
    """
    Function to calculate IoU (Intersection over Union) between two bounding boxes.
    Each bounding box is expected to be in the format:
    (xmin, ymin, xmax, ymax)
    """
    x1_topleft, y1_topleft, x1_bottomright, y1_bottomright = box1
    x2_topleft, y2_topleft, x2_bottomright, y2_bottomright = box2

    # Calculate intersection coordinates
    x_left = max(x1_topleft, x2_topleft)
    y_top = max(y1_topleft, y2_topleft)
    x_right = min(x1_bottomright, x2_bottomright)
    y_bottom = min(y1_bottomright, y2_bottomright)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = (x1_bottomright - x1_topleft) * (y1_bottomright - y1_topleft)
    box2_area = (x2_bottomright - x2_topleft) * (y2_bottomright - y2_topleft)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def is_true_positive(pred_box, gt_box, threshold=0.5):
    """
    Function to determine if a predicted bounding box is a True Positive based on IoU threshold.
    pred_box and gt_box should be in format (xmin, ymin, xmax, ymax).
    """
    iou = calculate_iou(pred_box, gt_box)
    if iou >= threshold:
        return True  # True Positive
    else:
        return False  # Not a True Positive
