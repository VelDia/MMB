from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov10/ultralytics/cfg/models/v10/yolov10n.yaml")  # load a pretrained model (recommended for training)
# model = YOLO("/home/diana/MMB/runs/detect/train8/weights/best.pt")
model = YOLO("/home/diana/MMB/weights/yolov10n.pt")
# Use the model
model.train(data="/home/diana/MMB/planes.yaml", epochs=50)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# path = model.export(format="onnx")  # export the model to ONNX format

results = model.predict(source='/home/diana/MMB/yolo_planes/train/images', save=True, project='/home/diana/MMB/voc_dataset/predictions_test_yolov10_pretr_planes')

# The results object contains the predictions for each image
# for result in results:
#     print(f"Processed {result.path}")
#     print(result.pandas().xyxy[0]) 

results.show()
results.save("results.txt")
    