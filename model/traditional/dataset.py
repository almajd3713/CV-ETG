import cv2
import torch
from torch.utils.data import Dataset

import os
import json
import hashlib

from transformers import AutoImageProcessor

with open("config.json", 'r') as f:
  config = json.load(f)

base_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_data_metadata = os.path.join("data", "labels.json")
synthetic_data_dir = os.path.join(base_dir, "synthetic_images")

with open(synthetic_data_metadata, 'r') as f:
    synthetic_data = json.load(f)

raw_image_data = os.path.join('data', config["locations"]["raw_images_metadata"])
with open(raw_image_data, 'r') as f:
  raw_image_data = json.load(f)

categories = raw_image_data['classes']
id2label = {index: x for index, x in enumerate(categories, start=1)}
label2id = {v: k for k, v in id2label.items()}

import albumentations as A
train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)
valid_test_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)
class TraditionalCNNDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def load_data(self):
        if not os.path.exists(synthetic_data_metadata):
            raise FileNotFoundError(f"Synthetic data metadata file not found: {synthetic_data_metadata}")
        with open(synthetic_data_metadata, 'r') as f:
            data = json.load(f)
        return data

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __len__(self):
        return len(self.data)
    
    def parse_image_id(self, id: str):
        """
        Turn uuid image id into int
        """
        return int(hashlib.sha1(id.encode()).hexdigest(), 16) % (10**8)

    def __getitem__(self, idx):
        """
        The getter here is for the image, thus loading all the associated bounding boxes and their classes
        """
        image_data = self.data[idx]
        image_path = image_data['path']
        image_bytes = self.load_image(image_path)
        bbox = image_data['bbox']
        bbox_arr = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]

        label = label2id[image_data['class']]
        # Apply my own transforms
        if self.transform:
            augmented = self.transform(image=image_bytes, bboxes=[bbox_arr], labels=[label])
            image_bytes = augmented['image']
            bboxes = augmented['bboxes']
            labels = augmented['labels']
        # Apply the pretrained transforms
        formatted_annotations = {
            "image_id": self.parse_image_id(image_data['image_id']),
            "annotations": [{"bbox": bbox, "category_id": label, "area": bbox[2] * bbox[3]} for bbox, label in zip(bboxes, labels)]
        }
        return {
            'pixel_values': image_bytes,
            'labels': torch.tensor(labels, dtype=torch.int64),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'image_id': formatted_annotations['image_id']
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding boxes and classes.
    """
    output = {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': [item['labels'] for item in batch],
        'bboxes': [item['bboxes'] for item in batch],
        'image_id': [item['image_id'] for item in batch]
    }
    return output
    
def train_valid_test_split(dataset_metadata, train_ratio=0.8, valid_ratio=0.1):
    """
    Splits the dataset into train, validation, and test sets.
    """
    total_size = len(set([image['image_id'] for image in dataset_metadata]))
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    indices = list(range(total_size))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train = [dataset_metadata[i] for i in train_indices]
    valid = [dataset_metadata[i] for i in valid_indices]
    test = [dataset_metadata[i] for i in test_indices]

    return train, valid, test
