from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov10/weights/yolov10x.pt")  # load a pretrained model (recommended for training)
model = YOLO("/home/diana/MMB/runs/detect/train4/weights/best.pt")
# Use the model
# model.train(data="/home/diana/MMB/viso.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# path = model.export(format="onnx")  # export the model to ONNX format

results = model.predict(source='/home/diana/MMB/voc_dataset/test/images', save=True, project='/home/diana/MMB/voc_dataset/predictions_test_yolov10')

# The results object contains the predictions for each image
# for result in results:
#     print(f"Processed {result.path}")
#     print(result.pandas().xyxy[0]) 

results.show()
    