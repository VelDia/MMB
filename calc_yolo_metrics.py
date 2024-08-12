import os
import csv

path = '/Users/diana/Desktop/MMB/mot/car/001/img/output_yolov5_orig'
txt_list = sorted(os.listdir(path))
txt_path = [os.path.join(path, file) for file in txt_list if file.endswith('.txt')]

def read_csv_file_yolo(files):
    """
    Reads a CSV file and returns a list of bounding boxes.
    Each line is expected to be in the format: xmin,ymin,xmax,ymax
    """
    bounding_boxes = []
    for file_path in files:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            # next(reader)  # Skip the header row
            i = 0
            bb_img =[]
            for row in reader:

                _, xmin, ymin, height, width = map(int, row)
                
                xmax = xmin + height
                ymax = ymin + width
                
                bb_img.append([xmin, ymin, xmax, ymax])
        bounding_boxes.append(bb_img)

    return bounding_boxes