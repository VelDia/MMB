from part3 import read_csv_file, calculate_metrics
import numpy as np

gt_path = 'mot/car/001/gt/gt.txt'
# pred_path = 'output_rois_new/car/1/pred_alg1.txt'
pred_path = 'output_rois_orig/pred_alg2.txt'

ground_truths = read_csv_file(gt_path)
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