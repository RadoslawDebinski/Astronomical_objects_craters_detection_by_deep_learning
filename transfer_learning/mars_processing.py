import contextlib
import math
import os
import zipfile
import shutil
import cv2
import pandas as pd
import numpy as np
from PIL import Image

from settings import CONST_PATH, MARS_TILE_DEG_SPAN, \
    MARS_CATALOGUE_NAME, MARS_CATALOGUE_LAT, MARS_CATALOGUE_LONG, MARS_CATALOGUE_DIAM, \
    MARS_SCALE_KM, MEAN_MARS_RADIUS_KM, LONGITUDE_MARS_CIRCUMFERENCE_KM, \
    CRATER_RIM_INTENSITY, SAMPLE_RESOLUTION, MARS_KERNEL_SIZE
from dataset_creation.dataset_creation_utils import dir_module


class MarsSamples:
    def __init__(self, no_samples):
        # Check directories
        dir_module()
        self.no_samples = no_samples
        # Data containers
        self.file_name = None
        self.src_image = None
        self.src_mask = None
        self.src_image_shape_0 = None
        self.src_image_shape_1 = None
        # Mask processing variables
        self.lat_min_limit = None
        self.lat_max_limit = None
        self.long_min_limit = None
        self.long_max_limit = None
        self.rim_intensity = CRATER_RIM_INTENSITY
        # Load mars crater catalogue
        self.catalogue = pd.read_csv(os.path.join(CONST_PATH["tempMars"], MARS_CATALOGUE_NAME), low_memory=False)

    @staticmethod
    def unzip_mars_tiles(open_zip=True, copy_tif=True):
        """
        Purpose of this script is to unzip files from data/temp/mars
        and copy .tif files from them to data/transfer_learning/original
        """
        # Check directories
        dir_module()
        # Unzip all zip files from temp Mars directory
        if open_zip:
            # List all the files in the folder
            file_list = os.listdir(CONST_PATH['tempMars'])

            # Iterate through the files and unzip the ones with a '.zip' extension
            for file_name in file_list:
                if file_name.endswith('.zip'):
                    file_path = os.path.join(CONST_PATH['tempMars'], file_name)
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        print(f'Unzipping {file_name}')
                        zip_ref.extractall(CONST_PATH['tempMars'])

        # Iterate through all unzipped directories and look for .tif files then copy them
        if copy_tif:
            # Use os.listdir to get a list of all files and folders in the specified directory
            contents = os.listdir(CONST_PATH['tempMars'])

            # Filter out only the folders (directories) and iterate
            for folder_name in [item for item in contents if os.path.isdir(os.path.join(CONST_PATH['tempMars'], item))]:
                for file in os.listdir(os.path.join(CONST_PATH["tempMars"], folder_name)):
                    if file.endswith('.tif'):
                        print(f'Copying {file} to {os.path.join(CONST_PATH["marsORG"], file)}')
                        shutil.copy(os.path.join(CONST_PATH["tempMars"], folder_name, file),
                                    os.path.join(CONST_PATH["marsORG"], file))

    def calc_bounds(self):
        lat_min_limit, long_min_limit = self.file_name.split("E")[1].split("N")
        self.lat_min_limit, self.long_min_limit = map(lambda x: int(x.split("_")[0]),
                                                      [lat_min_limit, long_min_limit])
        self.lat_max_limit, self.long_max_limit = map(lambda x: x + MARS_TILE_DEG_SPAN,
                                                      [self.lat_min_limit, self.long_min_limit])
        # Because of the standard for catalogue of craters where longitude has range from 0 to 360
        # we have to convert current variable which has range from -180 to 180
        self.long_min_limit = self.long_min_limit + 360 if self.long_min_limit <= 0 else self.long_min_limit
        self.long_max_limit = self.long_max_limit + 360 if self.long_max_limit <= 0 else self.long_max_limit

    def load_src(self):
        # Default size of image in Pillow cannot exceed nearly 1,8 * 10^8px
        # Our images has nearly 2,25 * 10^9px
        # So limit of pixels in Pillow have to be enlarged or for example deleted via line below
        Image.MAX_IMAGE_PIXELS = None
        self.src_image = np.array(Image.open(os.path.join(CONST_PATH["marsORG"], self.file_name))).astype(np.uint8)
        self.src_mask = np.zeros(np.shape(self.src_image)).astype(np.uint8)
        self.src_image_shape_0 = np.shape(self.src_image)[0]
        self.src_image_shape_1 = np.shape(self.src_image)[1]

    def create_mask(self):
        # Get all craters for this tile from catalogue
        filtered_rows = self.catalogue[
            (self.catalogue[MARS_CATALOGUE_LAT] > self.lat_min_limit) &
            (self.catalogue[MARS_CATALOGUE_LAT] < self.lat_max_limit) &
            (self.catalogue[MARS_CATALOGUE_LONG] > self.long_min_limit) &
            (self.catalogue[MARS_CATALOGUE_LONG] < self.long_max_limit)
            ]

        # Extract the second and third columns as lists
        latitude_list = filtered_rows[MARS_CATALOGUE_LAT].tolist()
        longitude_list = filtered_rows[MARS_CATALOGUE_LONG].tolist()
        diameters_list = filtered_rows[MARS_CATALOGUE_DIAM].tolist()

        num_rows = len(latitude_list)

        print(f"Processing craters for: \"{self.file_name}\" started.")
        process_counter = 0
        print(f"Craters placing: {process_counter}%", end='\r')

        radius_scaler = 1

        # Combine the lists into a list of tuples and draw every crater on mask image
        for latitude, longitude, diameter in list(zip(latitude_list, longitude_list, diameters_list)):
            radius_km = diameter / 2
            crater_circum_km = 2 * math.pi * radius_km
            steps = int(crater_circum_km / MARS_SCALE_KM)
            for step in range(steps):
                # Calculating pixels for rim with offset
                beta = 2 * math.pi * step / steps
                r_x = radius_scaler * radius_km * math.sin(beta)
                r_y = radius_scaler * radius_km * math.cos(beta)
                latitude_mars_circumference_km = math.sin(
                    math.pi / 2 - math.radians(latitude)) * 2 * math.pi * MEAN_MARS_RADIUS_KM
                gamma_x = math.degrees(r_x * 2 * math.pi / latitude_mars_circumference_km)
                gamma_y = math.degrees(r_y * 2 * math.pi / LONGITUDE_MARS_CIRCUMFERENCE_KM)
                pixel_x = int((longitude + gamma_x - self.long_min_limit) * self.src_image_shape_1 / MARS_TILE_DEG_SPAN)
                pixel_y = int((self.lat_max_limit - latitude - gamma_y) * self.src_image_shape_0 / MARS_TILE_DEG_SPAN)
                # Place a pixel at the specified coordinates
                with contextlib.suppress(IndexError):
                    # Create a circular mask
                    mask = np.zeros_like(self.src_mask, dtype=np.uint8)
                    cv2.circle(mask, (pixel_x, pixel_y), MARS_KERNEL_SIZE, self.rim_intensity, thickness=-1)

                    # # Update the original array using the mask
                    # self.src_mask = np.maximum(self.src_mask, mask)
                    # Update the original array using bitwise OR

                    self.src_mask = cv2.bitwise_or(self.src_mask, mask)
                print(f"Crater placing: {round(step / steps * 100)}%", end='\r')

            print(f"Craters placing: {round(process_counter / num_rows * 100)}%", end='\r')
            process_counter += 1
        # Process finished display summary
        print("Craters placing 100%")
        # # Create a kernel for dilation
        # kernel = np.ones((MARS_KERNEL_SIZE, MARS_KERNEL_SIZE), np.uint8)
        # # Dilate the white areas in second_mask
        # self.src_mask = cv2.dilate(self.src_mask, kernel, iterations=1)

    def create_dataset(self):
        # Every imported image will be split to 4 independent samples
        no_samples = self.no_samples / 4

        # TODO here will be added gathering list of .zip files available on MURRAY_LAB_URL

        # TODO here will be sorting of that list of .zip files via longitude

        # Samples creation loop starts here

        # TODO here will be added downloading package with image from MURRAY_LAB_URL

        # TODO here will be added unpacking image from downloaded zip

        # Here bounds calculation

        # TODO here will be mask creation for that image

        # TODO here will be sample creation

    def create_samples(self, show_examples=False):
        # Split the image into quadrants
        left_upper_src_img = cv2.resize(self.src_image[:self.src_image_shape_0 // 2, :self.src_image_shape_1 // 2],
                                        SAMPLE_RESOLUTION)
        right_upper_src_img = cv2.resize(self.src_image[:self.src_image_shape_0 // 2, self.src_image_shape_1 // 2:],
                                         SAMPLE_RESOLUTION)
        left_lower_src_img = cv2.resize(self.src_image[self.src_image_shape_0 // 2:, :self.src_image_shape_1 // 2],
                                        SAMPLE_RESOLUTION)
        right_lower_src_img = cv2.resize(self.src_image[self.src_image_shape_0 // 2:, self.src_image_shape_1 // 2:],
                                         SAMPLE_RESOLUTION)
        # Split the mask into quadrants
        left_upper_src_mask = cv2.resize(self.src_mask[:self.src_image_shape_0 // 2, :self.src_image_shape_1 // 2],
                                         SAMPLE_RESOLUTION)
        right_upper_src_mask = cv2.resize(self.src_mask[:self.src_image_shape_0 // 2, self.src_image_shape_1 // 2:],
                                          SAMPLE_RESOLUTION)
        left_lower_src_mask = cv2.resize(self.src_mask[self.src_image_shape_0 // 2:, :self.src_image_shape_1 // 2],
                                         SAMPLE_RESOLUTION)
        right_lower_src_mask = cv2.resize(self.src_mask[self.src_image_shape_0 // 2:, self.src_image_shape_1 // 2:],
                                          SAMPLE_RESOLUTION)
        # Safe input samples
        cv2.imwrite(os.path.join(CONST_PATH["marsIN"], "0" + self.file_name.replace(".tif", ".jpg")),
                    left_upper_src_img)
        cv2.imwrite(os.path.join(CONST_PATH["marsIN"], "1" + self.file_name.replace(".tif", ".jpg")),
                    right_upper_src_img)
        cv2.imwrite(os.path.join(CONST_PATH["marsIN"], "2" + self.file_name.replace(".tif", ".jpg")),
                    left_lower_src_img)
        cv2.imwrite(os.path.join(CONST_PATH["marsIN"], "3" + self.file_name.replace(".tif", ".jpg")),
                    right_lower_src_img)
        # Safe output samples
        cv2.imwrite(os.path.join(CONST_PATH["marsOUT"], "0" + self.file_name.replace(".tif", ".jpg")),
                    left_upper_src_mask)
        cv2.imwrite(os.path.join(CONST_PATH["marsOUT"], "1" + self.file_name.replace(".tif", ".jpg")),
                    right_upper_src_mask)
        cv2.imwrite(os.path.join(CONST_PATH["marsOUT"], "2" + self.file_name.replace(".tif", ".jpg")),
                    left_lower_src_mask)
        cv2.imwrite(os.path.join(CONST_PATH["marsOUT"], "3" + self.file_name.replace(".tif", ".jpg")),
                    right_lower_src_mask)

        if show_examples:
            self.show_example(left_upper_src_img, left_upper_src_mask)
            self.show_example(right_upper_src_img, right_upper_src_mask)
            self.show_example(left_lower_src_img, left_lower_src_mask)
            self.show_example(right_lower_src_img, right_lower_src_mask)

    @staticmethod
    def show_example(resized_input_image, resized_mask_image):
        combined_image = cv2.hconcat([resized_input_image, resized_mask_image])
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def example_temp(self):
        contents = os.listdir(CONST_PATH["marsORG"])
        self.file_name = contents[0]
        self.load_src()
        self.calc_bounds()
        self.create_mask()
        self.create_samples(True)
        print("End of example")
