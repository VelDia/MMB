import cv2
import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(model='runs/detect/train8/weights/best.pt')
gt_path_low_img = "voc_dataset/med/images/029128.jpg"
image = cv2.imread(gt_path_low_img)
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)