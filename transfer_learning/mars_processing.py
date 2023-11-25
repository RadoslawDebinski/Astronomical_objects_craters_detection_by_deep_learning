import contextlib
import math
import os
import re

import cv2
import pandas as pd
import numpy as np
from PIL import Image

from settings import CONST_PATH, MARS_TILE_DEG_SPAN, \
    MARS_CATALOGUE_NAME, MARS_CATALOGUE_LAT, MARS_CATALOGUE_LONG, MARS_CATALOGUE_DIAM, \
    MARS_SCALE_KM, MEAN_MARS_RADIUS_KM, LONGITUDE_MARS_CIRCUMFERENCE_KM, \
    CRATER_RIM_INTENSITY, MARS_KERNEL_SIZE, MARS_MASK_RESOLUTION, \
    MURRAY_LAB_URL, MAX_MARS_SAMPLES_BORDER

from dataset_creation.dataset_creation_utils import dir_module
from transfer_learning.mars_online_data_utils import get_zip_list_url, sort_tiles_longitude, \
    download_image


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
        self.catalogue = pd.read_csv(os.path.join(CONST_PATH["cataORG"], MARS_CATALOGUE_NAME), low_memory=False)

    def calc_bounds(self):
        long_min_limit, lat_min_limit = self.file_name.split("E")[1].split("N")
        self.lat_min_limit, self.long_min_limit = map(lambda x: int(x.split("_")[0]),
                                                      [lat_min_limit, long_min_limit])
        self.lat_max_limit, self.long_max_limit = map(lambda x: x + MARS_TILE_DEG_SPAN,
                                                      [self.lat_min_limit, self.long_min_limit])
        # Because of the standard for catalogue of craters where longitude has range from 0 to 360
        # we have to convert current variable which has range from -180 to 180
        self.long_min_limit = self.long_min_limit + 360 if self.long_min_limit <= 0 else self.long_min_limit
        self.long_max_limit = self.long_max_limit + 360 if self.long_max_limit <= 0 else self.long_max_limit

    def load_src(self, image_data):
        # Default size of image in Pillow cannot exceed nearly 1,8 * 10^8px
        # Our images has nearly 2,25 * 10^9px
        # So limit of pixels in Pillow have to be enlarged or for example deleted via line below
        Image.MAX_IMAGE_PIXELS = None
        self.src_image = cv2.resize(np.array(Image.open(image_data)).astype(np.uint8), MARS_MASK_RESOLUTION)
        self.src_mask = np.zeros(np.shape(self.src_image)).astype(np.uint8)
        self.src_image_shape_0 = np.shape(self.src_image)[0]
        self.src_image_shape_1 = np.shape(self.src_image)[1]

    def _mark_craters_rim(self, latitude_list, longitude_list, diameters_list):
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
                    self.src_mask[pixel_y, pixel_x] = self.rim_intensity

            print(f"Craters placing: {round(process_counter / num_rows * 100)}%", end='\r')
            process_counter += 1
        # Process finished display summary
        print("Craters placing 100%")

    def create_mask(self, add_kernel=False):
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

        self._mark_craters_rim(latitude_list, longitude_list, diameters_list)

        if add_kernel:
            # Create a kernel for dilation
            kernel = np.ones((MARS_KERNEL_SIZE, MARS_KERNEL_SIZE), np.uint8)
            # Dilate the white areas in second_mask
            self.src_mask = cv2.dilate(self.src_mask, kernel, iterations=1)

    def create_samples(self, show_examples=False):
        # Split the image into quadrants
        left_upper_src_img = self.src_image[:self.src_image_shape_0 // 2, :self.src_image_shape_1 // 2]
                                        
        right_upper_src_img = self.src_image[:self.src_image_shape_0 // 2, self.src_image_shape_1 // 2:]
                                         
        left_lower_src_img = self.src_image[self.src_image_shape_0 // 2:, :self.src_image_shape_1 // 2]
                                        
        right_lower_src_img = self.src_image[self.src_image_shape_0 // 2:, self.src_image_shape_1 // 2:]
                                         
        # Split the mask into quadrants
        left_upper_src_mask = self.src_mask[:self.src_image_shape_0 // 2, :self.src_image_shape_1 // 2]
                                         
        right_upper_src_mask = self.src_mask[:self.src_image_shape_0 // 2, self.src_image_shape_1 // 2:]
                                          
        left_lower_src_mask = self.src_mask[self.src_image_shape_0 // 2:, :self.src_image_shape_1 // 2]
                                         
        right_lower_src_mask = self.src_mask[self.src_image_shape_0 // 2:, self.src_image_shape_1 // 2:]
                                          
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

    def show_generated_files(self):
        for input_name in os.listdir(CONST_PATH["marsIN"]):
            if input_name in os.listdir(CONST_PATH["marsOUT"]):
                input_path = os.path.join(CONST_PATH['marsIN'], input_name)
                output_path = os.path.join(CONST_PATH['marsOUT'], input_name)
                print(f"Comparation of {input_path} and {output_path}")
                self.show_example(np.array(Image.open(input_path)), np.array(Image.open(output_path)))
            else:
                print(f"No output file found for {input_name}")

    def check_out_path(self, name):
        lat_min_limit, long_min_limit = name.split("E")[1].split("N")
        lat_min_limit = lat_min_limit[:-1]
        long_min_limit = long_min_limit.replace(".zip", "")
        r = re.compile(f"\dMurrayLab_CTX_V01_E{lat_min_limit}_N{long_min_limit}_Mosaic")
        return bool(list(filter(r.match, os.listdir(CONST_PATH["marsIN"]))))

    def create_dataset(self):
        # Number of samples cannot exceed max available amount of data
        if self.no_samples >= MAX_MARS_SAMPLES_BORDER:
            return print(f"Number of samples: {self.no_samples} is above "
                         f"the limit of: {MAX_MARS_SAMPLES_BORDER} samples.")

        # Every imported image will be split to 4 independent samples
        no_samples = int(self.no_samples / 4)
        # Gathering list of .zip files available on MURRAY_LAB_URL
        available_downloads = get_zip_list_url(MURRAY_LAB_URL)
        # Sorting that list of .zip files via longitude
        correct_downloads = sort_tiles_longitude(available_downloads)
        # Taking only the number of samples required by a user
        correct_downloads = dict(list(correct_downloads.items())[:no_samples])
        # Samples creation loop starts here
        for href in correct_downloads:
            print(f'Checking if file {href} was already downloaded.')
            if self.check_out_path(href):
                print(f'file {href} was already downloaded. Skip.')
                continue
            print(f'Downloading file: {href}')
            # Collect image name and data from href
            self.file_name, image_bit_stream = download_image(correct_downloads[href])
            if self.file_name is None or image_bit_stream is None:
                continue
            self.load_src(image_bit_stream)
            # Mask and samples creation process
            self.calc_bounds()
            self.create_mask()
            self.create_samples()

