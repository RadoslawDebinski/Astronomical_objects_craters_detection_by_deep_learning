from settings import KERNEL_SIZE

import contextlib
import cv2
import numpy as np
import pandas as pd
import math


class MaskCreator:
    def __init__(self, scale, moon_radius_km, long_moon_circum_km, rim_intensity):
        self.gray_img = None
        self.rgb_img = None
        self.scale = scale
        self.mask_img = None
        self.moon_radius_km = moon_radius_km
        self.long_moon_circum_km = long_moon_circum_km
        self.rim_intensity = rim_intensity

    def img_load(self, file_path):
        """
        Load image as grayscale one and prepare clear background for mask
        """
        self.gray_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        # Initiate mask image with gray image's shape
        self.mask_img = np.zeros(np.shape(self.gray_img))

    def img_analyze(self):
        """
        Print parameters of tile
        """
        height, width = self.gray_img.shape
        print("Image properties:")
        print(f"Height: {height} px")
        print(f"Width: {width} px")

    def _mark_crater_rim(self, longitude, latitude, long_limit, lat_limit, radius_km):
        """
        Function to draw crater's rim in given position and with defined radius
        """
        if radius_km > 0:
            # Draw circumference around center point
            crater_circum_km = 2 * math.pi * radius_km
            steps = int(crater_circum_km / self.scale)
            radius_scaler = 1

            for step in range(steps):
                # Calculating pixels for rim with offset
                beta = 2 * math.pi * step / steps
                r_x = radius_scaler * radius_km * math.sin(beta)
                r_y = radius_scaler * radius_km * math.cos(beta)
                latitude_moon_circumference_km = math.sin(
                    math.pi / 2 - math.radians(latitude)) * 2 * math.pi * self.moon_radius_km
                gamma_x = math.degrees(r_x * 2 * math.pi / latitude_moon_circumference_km)
                gamma_y = math.degrees(r_y * 2 * math.pi / self.long_moon_circum_km)
                pixel_x = int((longitude + gamma_x - long_limit) * self.gray_img.shape[1] / 90)
                pixel_y = int((lat_limit - latitude - gamma_y) * self.gray_img.shape[0] / 60)
                # Place a pixel at the specified coordinates
                with contextlib.suppress(IndexError):
                    self.mask_img[pixel_y, pixel_x] = self.rim_intensity

    def place_craters(self, csv_dir, bounds):
        """
        Simple pipeline to process craters from CSV sub-catalogue for selected WAC tile
        """
        # Read the CSV file with dataset into a Pandas DataFrame
        df = pd.read_csv(csv_dir)
        # Initiate process variables and communicates
        num_rows = df.shape[0]
        print(f"Processing craters from: \"{csv_dir}\" started.")
        process_counter = 0
        print(f"Craters placing: {process_counter}%", end='\r')
        rejected_craters_counter = 0
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Access the second and third columns by their names
            longitude = row['LON_CIRC_IMG']
            latitude = row['LAT_CIRC_IMG']
            # Check if crater center is in bounds
            if bounds[2] < longitude < bounds[3] and bounds[0] < latitude < bounds[1]:
                radius = row['DIAM_CIRC_IMG'] / 2
                self._mark_crater_rim(longitude, latitude, bounds[2], bounds[1], radius)
            else:
                rejected_craters_counter += 1
            # Update process status
            if int(index / num_rows * 100) > process_counter:
                process_counter = int(index / num_rows * 100)
                print(f"Craters placing: {process_counter}%", end='\r')

        # Create a kernel for dilation
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        # Dilate the white areas in second_mask
        self.mask_img = cv2.dilate(self.mask_img, kernel, iterations=1)

        # Process finished display summary
        print("Craters placing 100%")
        print(f"No. rejected craters: {rejected_craters_counter}, it's {rejected_craters_counter / num_rows}%")

    def save_mask(self, output_path):
        """
        Saving created mask for WAC tile
        """
        if cv2.imwrite(output_path, self.mask_img):
            print("Mask saved successfully.")
        else:
            print("Mask not saved successfully!")