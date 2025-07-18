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
synthetic_data_metadata = os.path.join(base_dir, "..", config["locations"]["labels"])
synthetic_data_dir = os.path.join(base_dir, "synthetic_images")

with open(synthetic_data_metadata, 'r') as f:
    synthetic_data = json.load(f)

import albumentations as A

train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomCrop(height=800, width=400),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
)
valid_test_transforms = A.Compose([A.NoOp()],
    bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
)

checkpoint = config['model']['pretrained_weights']

preprocessor = AutoImageProcessor.from_pretrained(
  checkpoint,
  do_resize=True,
  size=config['training']['train_input_size'],
)

class PytorchDataset(Dataset):
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
        return len(self.data['images'])
    
    def parse_image_id(self, id: str):
        """
        Turn uuid image id into int
        """
        return int(hashlib.sha1(id.encode()).hexdigest(), 16) % (10**8)

    def __getitem__(self, idx):
        """
        The getter here is for the image, thus loading all the associated bounding boxes and their classes
        """
        image_data = self.data['images'][idx]
        image_path = image_data['path']
        image_bytes = self.load_image(image_path)
        annotations = [ann for ann in self.data['annotations'] if ann['image_id'] == image_data['id']]
        bboxes = [
            [ann['bbox']['x1'], ann['bbox']['y1'], ann['bbox']['x2'] - ann['bbox']['x1'], ann['bbox']['y2'] - ann['bbox']['y1']]
            for ann in annotations
        ]
        labels = [ann['category_id'] for ann in annotations]
        # Apply my own transforms
        if self.transform:
            augmented = self.transform(image=image_bytes, bboxes=bboxes, labels=labels)
            image_bytes = augmented['image']
            bboxes = augmented['bboxes']
            labels = augmented['labels']
        # Apply the pretrained transforms
        formatted_annotations = {
            "image_id": self.parse_image_id(image_data['id']),
            "annotations": [{"bbox": bbox, "category_id": label, "area": bbox[2] * bbox[3]} for bbox, label in zip(bboxes, labels)]
        }
        preprocessed_image = preprocessor(image_bytes, annotations=formatted_annotations, return_tensors="pt")
        # Squeezing the batch dimension
        result = {k: v[0] for k, v in preprocessed_image.items()}
        result['bboxes'] = torch.tensor(bboxes, dtype=torch.float32)
        result['class_labels'] = torch.tensor(labels, dtype=torch.int64)
        return result

def collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding boxes and classes.
    """
    images = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    return {
        'pixel_values': torch.stack(images, dim=0),
        'labels': labels
    }
    
def train_valid_test_split(dataset_metadata, train_ratio=0.8, valid_ratio=0.1):
    """
    Splits the dataset into train, validation, and test sets.
    """
    total_size = len(dataset_metadata['images'])
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    indices = list(range(total_size))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train = {
        'images': [dataset_metadata['images'][i] for i in train_indices],
        'annotations': dataset_metadata['annotations'],
        'categories': dataset_metadata['categories']
    }
    valid = {
        'images': [dataset_metadata['images'][i] for i in valid_indices],
        'annotations': dataset_metadata['annotations'],
        'categories': dataset_metadata['categories']
    }
    test = {
        'images': [dataset_metadata['images'][i] for i in test_indices],
        'annotations': dataset_metadata['annotations'],
        'categories': dataset_metadata['categories']
    }
    return train, valid, test
