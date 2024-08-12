import os
import cv2
import numpy as np
from part3 import read_csv_file, read_csv_file_yolo, calculate_metrics

img = cv2.imread('mot/car/001/img/000001.jpg')
width, height, _ = img.shape

gt_path = 'mot/car/001/gt/gt.txt'
ground_truths = read_csv_file(gt_path)
# ground_truths_norm = [
#     [[(item - 0) / (width - 0) for item in sublist]
#     for sublist in lists] 
#     for lists in ground_truths 
# ]
# ground_truths = np.array(ground_truths_norm, dtype=object)
ground_truths = np.array(ground_truths, dtype=object)
print(ground_truths)
path = 'mot/car/001/img/output_yolov5_orig'
txt_list = sorted(os.listdir(path))
txt_path = [os.path.join(path, file) for file in txt_list if file.endswith('.txt')]


preds = read_csv_file_yolo(txt_path,width, height)
preds = np.array(preds, dtype=object)

eval = calculate_metrics(preds, ground_truths)
print(eval)