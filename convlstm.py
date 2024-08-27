# import torch
import tensorflow
import ultralytics
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Load the pretrained YOLO model
yolo_model = load_model('yolo_model.h5')

# Freeze all layers in the YOLO model
for layer in yolo_model.layers:
    layer.trainable = False

from keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, Input
from keras.models import Model
sequence_length = 3
img_size = 1454
# Create a new input layer to handle sequences
sequence_input = Input(shape=(sequence_length,) + img_size)

# Add the pretrained YOLO model as part of the new model
yolo_features = yolo_model(sequence_input)

# Add ConvLSTM layers
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(yolo_features)
x = BatchNormalization()(x)
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
x = BatchNormalization()(x)
num_anchors = 9
num_classes = 4
# Add detection head
output = Conv2D(filters=num_anchors*(num_classes+5), kernel_size=(1, 1), padding='same')(x)

# Define the new model
model = Model(inputs=sequence_input, outputs=output)


# Compile the model
model.compile(optimizer='adam', loss='your_loss_function', metrics=['accuracy'])

# Train the model on your custom dataset
model.fit(data="/home/diana/MMB/viso.yaml", epochs=10)

