import os
import requests
from bs4 import BeautifulSoup
import json
import logging
# Set up logging
base_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(base_dir, "download_images.log")
images_dir = os.path.join(base_dir, "images")
images_json_dir = os.path.join(base_dir, "images_data.json")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')


wiki_base_url = "https://enterthegungeon.fandom.com/wiki/"
url = "https://enterthegungeon.fandom.com/wiki/References"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

def get_page_content(url):
  response = requests.get(url)
  return BeautifulSoup(response.text, "html.parser")

def get_usable_tables(soup):
  all_tables = soup.find_all("table", {"class": "wikitable"})
  return [table for table in all_tables if len(list(table.descendants)) > 10] # Removes nested tables

def get_image_data(table):
  rows = table.find("tbody").find_all("tr", recursive=False)
  rows = rows[1:]  # Skip the header row
  image_data = []
  for row in rows:
    try:
      image_block = row.find_all("td", recursive=False)
      image_url = image_block[0].find("a")["href"]
      image_article = image_block[1].find("a").text.strip()
      image_name = image_block[1].find("a")["title"].strip()
      image_data.append({
        "web_url": image_url,
        "article": wiki_base_url + image_article,
        "name": image_name
      })
    except Exception as e:
      print("Error processing row:", e)
      print("Row content:", row)
  return image_data

def download_images(image_data_list):
  for image_data in image_data_list:
    try:
      response = requests.get(image_data["web_url"], )
      if response.status_code == 200:
        with open(f"{images_dir}/{image_data['name']}.png", "wb") as f:
          f.write(response.content)
        image_data["url"] = f"{images_dir}/{image_data['name']}.png"
        logging.info(f"Downloaded {image_data['name']}")
      else:
        logging.error(f"Failed to download {image_data['name']}: {response.status_code}")
    except Exception as e:
      logging.error(f"Error downloading {image_data['name']}: {e}")

from concurrent.futures import ThreadPoolExecutor
def parallel_download_images(images_data, thread_count=10):
  # split the image data into different lists for parallel processing
  images_data_list = [images_data[i::thread_count] for i in range(thread_count)]
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(download_images, image_data) for image_data in images_data_list]
    for future in futures:
      future.result()  # Wait for all downloads to complete

if __name__ == "__main__":
  logging.info("Starting to process the page.")
  soup = get_page_content(url)
  tables = get_usable_tables(soup)
  
  all_image_data = []
  for table in tables:
    logging.info(f"Processing table with {len(list(table.children))} rows.")
    image_data = get_image_data(table)
    all_image_data.extend(image_data)
  
  logging.info(f"Total images found: {len(all_image_data)}")
  
  for image in all_image_data:
    logging.info(f"Image Name: {image['name']}, URL: {image['web_url']}, Article: {image['article']}")
  
  logging.info("Downloading images from files...")
  parallel_download_images(all_image_data)
  logging.info("All images downloaded successfully.")
  logging.info("Saving image data to file...")
  with open(images_json_dir, "w") as f:
    json.dump(all_image_data, f, indent=4)
  logging.info("Image data saved successfully.")
  logging.info("Process completed.")