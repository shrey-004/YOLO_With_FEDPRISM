import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml

class YOLODataset(Dataset):
    def __init__(self, data_yaml, img_dir, label_dir, img_size=640):
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))

        # Convert BGR to RGB and copy to remove negative stride
        img = img[:, :, ::-1].copy()

        # Convert to CHW format (channels first)
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        # Load labels
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, ndmin=2).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5))

        labels = torch.tensor(labels, dtype=torch.float32)
        return img, labels


# ✅ keep this OUTSIDE the class (no indentation)
def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO datasets.
    Keeps images stacked but keeps labels in a list (variable number per image).
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, labels
