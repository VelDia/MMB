import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load LeYOLO model
model = YOLO("/Users/diana/Desktop/Group9/Code/LeYOLO/weights/LeYOLOSmall.pt")

bbox_dir = '/Users/diana/Desktop/MMB/output_leyolo/bbox'
annotated_dir = '/Users/diana/Desktop/MMB/output_leyolo/annot'
# Path(bbox_dir).mkdir(parents=True, exist_ok=True)
# Path(annotated_dir).mkdir(parents=True, exist_ok=True)
os.makedirs(bbox_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)

# image_dir = '/Users/diana/Desktop/Group9/Data/Foggy_Cityscapes/test_original'#'/Users/diana/Desktop/Group9/Data/foggyX_images/test'#'/Users/diana/Desktop/Group9/Data/Foggy_Cityscapes/test_original'#'/Users/diana/Desktop/Group9/Data/foggyX_images/test'
# image_dir = '/Users/diana/Desktop/Group9/Data/Foggy_Cityscapes/test_original'
image_dir = '/Users/diana/Desktop/MMB/mot/car/001/img/resized_folder'

def process_and_save_images(image_dir):
    for img_path in Path(image_dir).glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            results = model(img_path)
            # Process detections
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                
                # Save bounding boxes to file
                with open(f'{bbox_dir}/{img_path.stem}.txt', 'w') as f:
                    for box in boxes:
                        cls = int(box.cls)
                        conf = box.conf.item()
                        xyxy = box.xyxy[0].tolist()
                        f.write(f'{model.names[cls]} {(*xyxy, conf)}\n')
                
                # Draw bounding boxes on image
                annotated_img = result.plot()
                
                # Save annotated image
                annotated_path = str(Path(annotated_dir) / img_path.name)
                cv2.imwrite(annotated_path, annotated_img)
    print("loaded images")

# Run the processing on the dehazed images directory
process_and_save_images(image_dir)
