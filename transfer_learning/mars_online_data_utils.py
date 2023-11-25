import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import urllib3
import warnings
import zipfile
from io import BytesIO

from settings import MAX_MARS_PROCESSING_LONGITUDE, URL_BACKUP_COUNTER_PATH


def get_zip_list_url(url):
    downloadable_files = {}
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all anchor tags (links) that point to downloadable files
        links = soup.find_all('a', href=True)

        # Extract and print the href attribute of each link
        for link in links:
            href = link.get('href')
            if href.endswith('.zip'):
                downloadable_files[href] = urljoin(url, href)

    else:
        print('Connection with page failed. Error code:', response.status_code)

    return downloadable_files


def sort_tiles_longitude(files_dict):
    approved_dict = {}
    for file_name in files_dict.keys():
        max_longitude = file_name[:-4].split("N")[1]
        if abs(int(max_longitude)) <= MAX_MARS_PROCESSING_LONGITUDE:
            approved_dict[file_name] = files_dict[file_name]

    return approved_dict


def download_image(url):
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
    print("Downloading Image")
    response = requests.get(url, verify=False)
    content_type = response.headers.get('Content-Type')
    if content_type == 'application/zip':
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                if first_tif_file := next(
                    (file for file in file_list if file.endswith('.tif')), None
                ):
                    # Extract the content of the first .tif file
                    with zip_ref.open(first_tif_file) as tif_file:
                        # Read the content of the .tif file into a BytesIO object
                        image_data = BytesIO(tif_file.read())

                        # For example, if you want to verify you have the data:
                        print("Content of the .tif file downloaded")  # Reading the bytes
                        return first_tif_file.split('/')[1], image_data
                else:
                    print("No .tif file found in the zip archive")
                    return None, None
        else:
            print("Failed to fetch the ZIP file")
            return None, None
    else:
        print("File is not the ZIP file")
        return None, None

