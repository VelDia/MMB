from yolov10.ultralytics.models import YOLOv10
# Load a pre-trained YOLOv10 model
model = YOLOv10("/home/diana/MMB/runs/detect/train8/weights/best.pt")

# Perform object detection on an image
results = model.track('/home/diana/MMB/vidoes/output_video.mp4', save=True, show=True, conf=0.15, tracker="botsort.yaml")
results = model.track('/home/diana/MMB/vidoes/output_video.mp4', save=True, show=True, conf=0.15, tracker="bytetrack.yaml")
results = model.track('/home/diana/MMB/vidoes/output_video.mp4', save=True, show=True, conf=0.15)

# # Save results in MOTChallenge format (frame, id, bbox, conf)
# with open('track_results.txt', 'w') as f:
#     for frame_id, result in enumerate(results):
#         for box in result.boxes:
#             bbox = box.xyxy[0].tolist()  # Convert from tensor to list
#             track_id = box.id.item()  # Get track id
#             conf = box.conf.item()  # Get confidence score
#             f.write(f'{frame_id+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')