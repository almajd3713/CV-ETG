import ultralytics
import os
import json

with open('config.json', 'r') as f:
  config = json.load(f)

pretrained_weights = config['model']['pretrained_weights']
pretrained_path = os.path.join('model', 'weights', pretrained_weights)

model = ultralytics.YOLO(pretrained_path)
