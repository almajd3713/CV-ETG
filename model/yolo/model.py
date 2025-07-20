import ultralytics
import os
import json

with open('config.json', 'r') as f:
  config = json.load(f)

model = ultralytics.YOLO(config['model']['pretrained_weights'])
