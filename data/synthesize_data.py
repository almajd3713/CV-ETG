import os
import json
import logging
import random
import cv2
import uuid

#! --- SETUP ---
base_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(base_dir, "logs", "synthesize_data.log")
if not os.path.exists(os.path.join(base_dir, "logs")):
    os.makedirs(os.path.join(base_dir, "logs"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')
#! --------------

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")

class DataSynthesizer:
  def __init__(self, data_file, backgrounds_file, max_frequency=100):
    self.data_file = data_file
    self.backgrounds_file = backgrounds_file
    self.classes, self.image_data = self.load_data()
    self.max_frequency = max_frequency
   
  def load_data(self):
    if not os.path.exists(self.data_file):
      logging.error(f"Data file {self.data_file} does not exist.")
      return []
    with open(self.data_file, 'r') as f:
      file = json.load(f)
      classes = file.get("classes", [])
      classes = [(classs, 0) for classs in classes if classs]  # (class, current frequency)
      images = file.get("images", [])
      return classes, images
  
  def load_backgrounds(self):
    if not os.path.exists(self.backgrounds_file):
      logging.error(f"Backgrounds file {self.backgrounds_file} does not exist.")
      return []
    with open(self.backgrounds_file, 'r') as f:
      backgrounds = json.load(f)
      return backgrounds

  def load_image(self, image_path, image_name = '', scale=False):
    if not os.path.exists(image_path):
      logging.error(f"Image {image_name} does not exist at {image_path}.")
      return None
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
      logging.error(f"Failed to load image {image_name}.")
      return None
    if scale:
      # Scale with interpolation and respect aspect ratio
      h, w = image.shape[:2]
      scale_factor = 4
      image = cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_NEAREST)
    return image

  def get_used_classes(self):
    not_full_classes = [classs for classs in self.classes if classs[1] < self.max_frequency]
    most_frequent = max(not_full_classes, key=lambda x: x[1])
    least_frequent = min(not_full_classes, key=lambda x: x[1])
    return most_frequent, least_frequent

  def get_underfilled_classes(self):
    underfilled_classes = [classs for classs, freq in self.classes if freq < self.max_frequency]
    return underfilled_classes

  def load_classes_sample(self, max_sample_size=5):
    sample_classes = self.get_underfilled_classes()
    sample_size = min(len(sample_classes), max_sample_size)
    sample_classes = random.sample(sample_classes, sample_size)
    sample = [it for it in self.image_data if it['class_name'] in sample_classes]
    return sample, [self.load_image(image['url'], image['name'], scale=True) for image in sample]

  def load_background_sample(self):
    sample = random.sample(self.backgrounds, 1)
    return sample, self.load_image(sample[0]['filename'])

  def update_frequency(self, classes):
    for i, (cls, freq) in enumerate(self.classes):
      if cls == classes:
        self.classes[i] = (cls, freq + 1)
        break
  
  def overlay_images(self, foreground, background):
    if foreground is None or background is None:
      return None
    h, w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    if h > bg_h or w > bg_w:
      logging.warning("Foreground image is larger than background image. Resizing foreground.")
      foreground = cv2.resize(foreground, (bg_w, bg_h))
    x_offset = random.randint(0, bg_w - w)
    y_offset = random.randint(0, bg_h - h)
    # Handle alpha stuff
    alpha = foreground[:, :, 3] / 255.0
    foreground_rgb = foreground[:, :, :3]
    # Get the region of interest from background
    roi = background[y_offset:y_offset+h, x_offset:x_offset+w]
    # Blend the images
    for c in range(0, 3):
      roi[:, :, c] = (foreground_rgb[:, :, c] * alpha + 
        roi[:, :, c] * (1 - alpha))
    # Put the blended ROI back
    background[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    return background, {
      'x1': x_offset,
      'y1': y_offset,
      'x2': x_offset + w,
      'y2': y_offset + h,
    }
  
  def synthesize(self):
    image_data_list = []
    _, background = self.load_background_sample()
    classes, foregrounds = self.load_classes_sample()
    image_name = uuid.uuid4().hex
    image_path = os.path.join(base_dir, "synthesized_images", f"{image_name}.png")
    # synthesized_image, bbox = self.overlay_images(foreground, background)
    final_image = background.copy()
    for image_data, foreground in zip(classes, foregrounds):
      if foreground is None:
        print(image_data)
      self.update_frequency(image_data['class_name'])
      synthesized_image, bbox = self.overlay_images(foreground, final_image)
      if synthesized_image is None:
        logging.error("Failed to synthesize image. Skipping.")
        continue
      image_data_list.append({
        "name": image_name,
        "class_name": image_data['class_name'],
        "path": image_path,
        "bbox": bbox,
      })
      final_image = synthesized_image
    if final_image is not None:
      with open(image_path, "wb") as f:
        cv2.imwrite(f.name, final_image)
      logging.info(f"Synthesized and saved image {image_name}")
    return image_data_list
  
  def synthesize_until_satisfied(self):
    image_data_list = []
    while len(self.get_underfilled_classes()):
      underfilled_length = len(self.get_underfilled_classes())
      most_frequent, least_frequent = self.get_used_classes()
      logging.info(f"Underfilled classes: {underfilled_length}")
      logging.info(f"Most frequent class: {most_frequent}")
      logging.info(f"Least frequent class: {least_frequent}")
      image_data_list.extend(self.synthesize())
    return image_data_list


if __name__ == "__main__":
  data_file = os.path.join(base_dir, "raw_images_data.json")
  backgrounds_file = os.path.join(base_dir, "background_images", "background_info.json")

  synthesizer = DataSynthesizer(data_file, backgrounds_file, max_frequency=5)
  logging.info("Loading classes and images...")
  synthesizer.backgrounds = synthesizer.load_backgrounds()
  if not synthesizer.backgrounds:
    logging.error("No backgrounds loaded. Exiting.")
    exit(1)
  
  sample_classes, sample_images = synthesizer.load_classes_sample()
  sample_backgrounds, background_image = synthesizer.load_background_sample()
  
  if not len(sample_classes) or not len(background_image):
    logging.error("No images or backgrounds loaded. Exiting.")
    exit(1)
  logging.info(f"Loaded {len(sample_classes)} classes and {len(sample_images)} images.")
  
  logging.info(f'Synthesizing images with {len(sample_images)} sample images and background {sample_backgrounds[0]["filename"]}')
  if not os.path.exists(os.path.join(base_dir, "synthesized_images")):
    os.makedirs(os.path.join(base_dir, "synthesized_images"))
  clear_directory(os.path.join(base_dir, "synthesized_images"))
  synthesized_data = synthesizer.synthesize_until_satisfied()
  with open(os.path.join(base_dir, "synthesized_images", "synthesized_data.json"), 'w') as f:
    json.dump(synthesized_data, f, indent=4)
  final_image_count = len(os.listdir(os.path.join(base_dir, "synthesized_images"))) - 1  # -1 for the synthesized_data.json file
  logging.info(f'Finished synthesizing images. Generated {len(synthesized_data)} items in {final_image_count} images.')