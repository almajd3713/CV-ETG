import json
import os
import logging
import cv2


with open("config.json", 'r') as f:
  config = json.load(f)
base_dir = os.path.dirname(os.path.abspath(__file__))
synthesized_metadata = os.path.join(base_dir, config["locations"]["synthetic_data_metadata"])
output_labels = os.path.join(config["locations"]["labels"])

raw_image_data = os.path.join(base_dir, config["locations"]["raw_images_metadata"])
with open(raw_image_data, 'r') as f:
  raw_image_data = json.load(f)
  classes = raw_image_data['classes']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(base_dir, "logs", "generate_labels.log"), filemode='w')
output_mode = config['model']['label_mode']

class LabelGenerator:
    def __init__(self, metadata_file, output_file, mode=output_mode):
        self.metadata_file = metadata_file
        self.output_file = output_file
        self.mode = mode

    def generate_labels(self):
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        logging.info(f"Generating labels from {self.metadata_file} in {self.mode} mode.")
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} items from metadata file.")
        labels = []
        if self.mode == 'pytorch':
          for item in data:
            logging.info(f"Processing item: {item['name']}")
            labels.append(self.generate_pytorch_representation(item))
        elif self.mode == 'coco':
            labels = {"categories": [], "annotations": [], "images": []}
            for item in data:
                category, annotation, image = self.generate_coco_representation(item)
                if not category['id'] in [cat['id'] for cat in labels["categories"]]:
                  labels["categories"].append(category)
                labels["annotations"].append(annotation)
                labels["images"].append(image)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        logging.info(f"Generated label for item: {item['name']}")

        with open(self.output_file, 'w') as f:
            json.dump(labels, f, indent=4)
        
        logging.info(f"Labels generated and saved to {self.output_file}")
    
    def generate_pytorch_representation(self, item):
      return {
        "image_id": item['name'] + '.png',
        "class": item['class_name'],
        "bbox": item['bbox'],
        "path": item['path'],
      }
    def generate_coco_representation(self, item):          
        category = {
          "id": classes.index(item['class_name']) + 1,
          "name": item['class_name'],
        }
        width = item['bbox']['x2'] - item['bbox']['x1']
        height = item['bbox']['y2'] - item['bbox']['y1']
        annotation = {
          "image_id": item['name'],
          "category_id": classes.index(item['class_name']) + 1,
          "bbox": item['bbox'],
          "area": width * height,
        }
        image = {
          "id": item['name'],
          "width": config["training"]["input_size"]["width"],
          "height": config["training"]["input_size"]["height"],
          "path": os.path.join(base_dir, config["locations"]["synthesize_data"], item['name'] + '.png'),
        }
        return category, annotation, image
      
if __name__ == "__main__":
    label_generator = LabelGenerator(synthesized_metadata, output_labels)
    label_generator.generate_labels()
    logging.info("Label generation completed successfully.")