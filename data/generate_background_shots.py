import os
import cv2
import logging
import random


base_dir = os.path.dirname(os.path.abspath(__file__))
background_videos_dir = os.path.join(base_dir, "background_videos")
background_images_dir = os.path.join(base_dir, "background_images")
if not os.path.exists(background_videos_dir):
    os.makedirs(background_videos_dir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(base_dir, "logs", "generate_background_shots.log"), filemode='w')

def load_videos(directory):
    video_files = [os.path.join(directory, f)
                   for f in os.listdir(directory)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return video_files
  

def get_random_screenshots(video, count=10):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    screenshots = []
    if total_frames <= 0 or count <= 0:
      return screenshots

    frame_indices = random.sample(range(total_frames), min(count, total_frames))
    for frame_idx in frame_indices:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
      ret, frame = cap.read()
      if ret:
        screenshots.append(frame)

    cap.release()
    return screenshots
def get_random_screenshots_batch(videos, count=10):
    all_screenshots = []
    for video in videos:
        screenshots = get_random_screenshots(video, count)
        all_screenshots.extend(screenshots)
    return all_screenshots
  
def save_screenshots(screenshots, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, screenshot in enumerate(screenshots):
        screenshot_filename = os.path.join(output_dir, f"screenshot_{idx}.png")
        cv2.imwrite(screenshot_filename, screenshot)
        print(f"Saved screenshot: {screenshot_filename}")

if __name__ == "__main__":
  logging.info("Starting to process background videos.")
  logging.info(f"Looking for videos in: {background_videos_dir}")
  video_files = load_videos(background_videos_dir)
  logging.info(f"Found {len(video_files)} videos.")
  logging.info("Generating random screenshots from videos.")
  screenshots = get_random_screenshots_batch(video_files, count=10)
  logging.info(f"Generated {len(screenshots)} screenshots.")
  logging.info("Saving screenshots to output directory.")
  save_screenshots(screenshots, background_images_dir)
  logging.info("All screenshots saved successfully.")
  logging.info("Process completed.")