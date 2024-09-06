from yolov10.ultralytics.models import YOLOv10
# Load a pre-trained YOLOv10 model
model = YOLOv10("/home/diana/MMB/runs/detect/train8/weights/best.pt")

# Perform object detection on an image
results = model.track('/home/diana/MMB/vidoes/output_video.mp4', save=True, show=True, conf=0.15)