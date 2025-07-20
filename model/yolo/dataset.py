import os
import json
import shutil

with open('config.json', 'r') as f:
  config = json.load(f)

base_dir = os.path.dirname(os.path.abspath(__file__))
yolo_dir = os.path.join(base_dir, "dataset")

images_names = [
  img.split('.')[0]
  for img in sorted([
    f for f in os.listdir(os.path.join("data", "yolo", "images"))
    if os.path.isfile(os.path.join("data", "yolo", "images", f))
  ])
]

raw_image_data = os.path.join('data', config["locations"]["raw_images_metadata"])
with open(raw_image_data, 'r') as f:
  raw_image_data = json.load(f)

categories = raw_image_data['classes']
id2label = {index: x for index, x in enumerate(categories, start=1)}
label2id = {v: k for k, v in id2label.items()}
  
def separate_yolo_dir():
  """
  Generate yolo directory structure if it doesn't exist
  Create train/val/test datasets and copy their files over
  """
  old_yolo_image_dir = os.path.join("data", "yolo", "images")
  old_yolo_label_dir = os.path.join("data", "yolo", "labels")
  if not os.path.exists(yolo_dir):
    os.makedirs(yolo_dir)
  # Create train/val/test splits
  if not os.path.exists(os.path.join(yolo_dir, "images")):
    os.makedirs(os.path.join(yolo_dir, "images", "train"))
    os.makedirs(os.path.join(yolo_dir, "images", "val"))
    os.makedirs(os.path.join(yolo_dir, "images", "test"))
  if not os.path.exists(os.path.join(yolo_dir, "labels")):
    os.makedirs(os.path.join(yolo_dir, "labels", "train"))
    os.makedirs(os.path.join(yolo_dir, "labels", "val"))
    os.makedirs(os.path.join(yolo_dir, "labels", "test"))
  # Copy images and labels to train/val/test splits
  splits = config['training']['splits']
  splits = {
    'train': int(len(images_names) * splits['train']),
    'val': int(len(images_names) * splits['val']),
    'test': int(len(images_names) * splits['test']),
  }
  train_names = images_names[:splits['train']]
  val_names = images_names[splits['train']:splits['train']+splits['val']]
  test_names = images_names[splits['train']+splits['val']:]
  for name in train_names:
    shutil.copy(os.path.join(old_yolo_image_dir, name + '.png'), os.path.join(yolo_dir, "images", "train", name + '.png'))
    shutil.copy(os.path.join(old_yolo_label_dir, name + '.txt'), os.path.join(yolo_dir, "labels", "train", name + '.txt'))
  for name in val_names:
    shutil.copy(os.path.join(old_yolo_image_dir, name + '.png'), os.path.join(yolo_dir, "images", "val", name + '.png'))
    shutil.copy(os.path.join(old_yolo_label_dir, name + '.txt'), os.path.join(yolo_dir, "labels", "val", name + '.txt'))
  for name in test_names:
    shutil.copy(os.path.join(old_yolo_image_dir, name + '.png'), os.path.join(yolo_dir, "images", "test", name + '.png'))
    shutil.copy(os.path.join(old_yolo_label_dir, name + '.txt'), os.path.join(yolo_dir, "labels", "test", name + '.txt'))


import yaml
def generate_data_yaml():
  """
  Generate data.yaml file for YOLOv11
  """
  data = {
    'train': os.path.join(yolo_dir, "images", "train"),
    'val': os.path.join(yolo_dir, "images", "val"),
    'nc': len(categories),
    'names': {i: name for i, name in enumerate(categories)},
  }
  with open(os.path.join(yolo_dir, "data.yaml"), 'w') as f:
    yaml.dump(data, f)


if __name__ == "__main__":
  separate_yolo_dir()
  generate_data_yaml()
  print(f"YOLO directory structure created at {yolo_dir} with data.yaml file.")
  print("You can now train your YOLO model using the generated data.")