# torch_convLSTM
import torch
from ultralytics import YOLO
import torch.nn as nn
from yolov10 import YOLOv10

import torch
import torch.nn as nn

num_anchors = 9
num_classes = 4

class ConvLSTM2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, return_sequences):
        super(ConvLSTM2D, self).__init__()
        self.return_sequences = return_sequences
        self.padding = padding
        
        self.conv_lstm = nn.LSTM(input_size=in_channels, hidden_size=out_channels, num_layers=1, batch_first=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        # Reshape x to (batch, sequence, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size, seq_len, channels * height * width)
        
        # Pass through LSTM
        x, _ = self.conv_lstm(x)
        
        if self.return_sequences:
            x = x.view(batch_size, seq_len, -1, height, width)
        else:
            x = x[:, -1, :, :, :]
        
        # Apply Conv2D
        x = self.conv(x)
        return x


class YOLOWithConvLSTM(nn.Module):
    def __init__(self, yolo_model, num_anchors, num_classes):
        super(YOLOWithConvLSTM, self).__init__()
        self.yolo_model = yolo_model
        
        # Define ConvLSTM layers
        self.convlstm1 = ConvLSTM2D(in_channels=3, out_channels=64, kernel_size=(3, 3), padding='same', return_sequences=True)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.convlstm2 = ConvLSTM2D(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same', return_sequences=True)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        # Define detection head
        self.detection_head = nn.Conv2d(in_channels=64, out_channels=num_anchors*(num_classes+5), kernel_size=(1, 1), padding='same')
    
    def forward(self, x):
        # Pass through YOLO model
        x = self.yolo_model(x)
        
        # Pass through ConvLSTM layers
        x = self.convlstm1(x)
        x = self.batchnorm1(x)
        x = self.convlstm2(x)
        x = self.batchnorm2(x)
        
        # Pass through detection head
        x = self.detection_head(x)
        return x

# Initialize your YOLO model (assume it's defined elsewhere)
yolo_model = YOLOv10()

# Load the pretrained weights
yolo_model.load_state_dict(torch.load('/home/diana/MMB/weights/yolov10x.pt'))


# Freeze all layers
for param in yolo_model.parameters():
    param.requires_grad = False

# Instantiate the model
model = YOLOWithConvLSTM(yolo_model, num_anchors, num_classes)

import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
# Training loop

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class YOLODataset(Dataset):
    def __init__(self, img_folder, label_folder, transform=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(img_path).convert('RGB')
        boxes = self._load_labels(label_path)

        if self.transform:
            image = self.transform(image)

        return image, boxes

    def _load_labels(self, label_path):
        if not os.path.exists(label_path):
            return torch.tensor([])  # No annotations for this image

        boxes = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert YOLO format to bounding box coordinates (top-left, bottom-right)
                xmin = center_x - width / 2
                xmax = center_x + width / 2
                ymin = center_y - height / 2
                ymax = center_y + height / 2

                boxes.append([class_id, xmin, ymin, xmax, ymax])
        
        return torch.tensor(boxes, dtype=torch.float32)

# Usage example
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = YOLODataset(img_folder='vocrs_dataset/train/images', label_folder='vocrs_dataset/train/labels', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images, targets = batch
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# pip install pytorch-convlstm

# import torch
# import torch.nn as nn
# from convlstm import ConvLSTM

# class YOLOWithConvLSTM(nn.Module):
#     def __init__(self, yolo_model, num_anchors, num_classes):
#         super(YOLOWithConvLSTM, self).__init__()
#         self.yolo_model = yolo_model

#         # Define ConvLSTM layers
#         self.convlstm1 = ConvLSTM(input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), padding=1, return_sequences=True, return_cell=False)
#         self.batchnorm1 = nn.BatchNorm2d(64)
#         self.convlstm2 = ConvLSTM(input_dim=64, hidden_dim=[64], kernel_size=(3, 3), padding=1, return_sequences=True, return_cell=False)
#         self.batchnorm2 = nn.BatchNorm2d(64)

#         # Define detection head
#         self.detection_head = nn.Conv2d(in_channels=64, out_channels=num_anchors*(num_classes+5), kernel_size=(1, 1), padding='same')

#     def forward(self, x):
#         # Pass through YOLO model
#         x = self.yolo_model(x)

#         # Pass through ConvLSTM layers
#         x, _ = self.convlstm1(x)
#         x = self.batchnorm1(x)
#         x, _ = self.convlstm2(x)
#         x = self.batchnorm2(x)

#         # Pass through detection head
#         x = self.detection_head(x)
#         return x
