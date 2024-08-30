#!/usr/bin/env python3

'''
Title: YOLOrs: Object Detection in Multimodal RemoteSensing Imagery
Authors: Manish Sharma, Mayur Dhanaraj, Srivallabha Karnam, Dimitris G. Chachlakis, Raymond Ptucha, Panos P. Markopoulos, Eli Saber
Date: 
URL: 
'''

import glob
import random
import os
import sys
import numpy as np
# from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip, vertical_flip, jumble_up, rotate, photometric
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=512):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


# class ListDataset(Dataset):
#     def __init__(self, list_path, img_size=512, augment=True, multiscale=True, normalized_labels=True):
#         with open(list_path, "r") as file:
#             self.img_files = file.readlines()

#         self.label_files = [path.replace("images", "labels").
#                             replace(".png", ".txt").replace(".jpg", ".txt") 
#                             for path in self.img_files]
#         self.img_size = img_size
#         self.max_objects = 100
#         self.augment = augment
#         self.multiscale = multiscale
#         self.normalized_labels = normalized_labels
#         self.min_size = self.img_size - 3 * 32
#         self.max_size = self.img_size + 3 * 32
#         self.batch_count = 0

class ListDataset(Dataset):
    def __init__(self, folder = '/home/diana/MMB/YOLORS/vocrs_dataset/train', img_size=512, augment=True, multiscale=True, normalized_labels=True):
        self.img_folder = img_folder = os.path.join(folder, 'images')
        self.label_folder = label_folder = os.path.join(folder, 'labels')
        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        
        # List all image files in the directory
        self.img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg')])
        # frames = [cv2.imread(image_file) for image_file in image_files]
        # self.img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg'))]
        
        # Generate corresponding label file paths
        self.label_files = [os.path.join(label_folder, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in self.img_files]
        
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
    
    # def __getitem__(self, index):
    #     # Load image and labels as before
    #     img_path = self.img_files[index % len(self.img_files)]
    #     label_path = self.label_files[index % len(self.img_files)]

    #     img_rgb_path = img_path.replace('images', 'images_rgb')
    #     img_ir_path = img_path.replace('images', 'images_ir')
        
    #     img_rgb = cv2.imread(img_rgb_path)
    #     img_ir = cv2.imread(img_ir_path)
    #     img = np.concatenate((img_rgb, img_ir), axis=2)
    #     img = img[:, :, (2, 1, 0, 3)]  # BGRIII -> RGBIII
        
    #     if self.augment:
    #         img = photometric(img)
            
    #     img = transforms.ToTensor()(img)
        
    #     if len(img.shape) != 3:
    #         img = img.unsqueeze(0)
    #         img = img.expand((3, img.shape[1:]))
        
    #     # _, h, w = img.shape
    #     w, h, _ = img.shape
    #     h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
    #     img, pad = pad_to_square(img, 0)
    #     _, padded_h, padded_w = img.shape
        
    #     targets = None
    #     if os.path.exists(label_path):
    #         boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
    #         x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    #         y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    #         x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    #         y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
    #         x1 += pad[0]
    #         y1 += pad[2]
    #         x2 += pad[1]
    #         y2 += pad[3]
    #         boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    #         boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    #         boxes[:, 3] *= w_factor / padded_w
    #         boxes[:, 4] *= h_factor / padded_h

    #         targets = torch.zeros((len(boxes), 7))
    #         targets[:, 1:] = boxes
        
    #     if self.augment:
    #         if np.random.random() < 0.5:
    #             img, targets = horisontal_flip(img, targets)
    #         if np.random.random() < 0.5:
    #             img, targets = vertical_flip(img, targets)
    #         if np.random.random() < 0.5:
    #             img, targets = rotate(img, targets)
    #         if np.random.random() < 0.5:
    #             img, targets = jumble_up(img, targets)

    #     return img_path, img, targets

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # img_path_rgb= img_path.replace('images_rgb','images_rgb')
        img_path_rgb = img_path
        img_path_ir= img_path.replace('images_rgb','images_ir')

        # img_rgb= Image.open(img_path_rgb)
        # img_ir= Image.open(img_path_ir)
        img_rgb= cv2.imread(img_path_rgb)
        img_ir= cv2.imread(img_path_ir)
        img= np.concatenate((img_rgb,img_ir),axis=2)
        img= img[:,:, (2, 1, 0, 3)]  # BGRIII -> RGBIII
        
        if self.augment:
            img = photometric(img)
            
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2) # taking the central x axis and width divided by 2
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 7))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            if np.random.random() < 0.5:
                img, targets = vertical_flip(img, targets)
            if np.random.random() < 0.5:
                img, targets = rotate(img, targets)
            if np.random.random() < 0.5:
                img, targets = jumble_up(img, targets)
        return img_path, img, targets
    

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

anyvar = ListDataset()