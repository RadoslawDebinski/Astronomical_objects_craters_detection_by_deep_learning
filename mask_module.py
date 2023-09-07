import contextlib
import cv2
import numpy as np
import pandas as pd
import math

# !!!MOON CONSTANTS!!!
MEAN_MOON_RADIUS_KM = 1737.05
LONGITUDE_MOON_CIRCUMFERENCE_KM = 10907
# Other variables
RIM_INTENSITY = 255


class MaskCreator:
    def __init__(self, scale):
        self.gray_img = None
        self.rgb_img = None
        self.scale = scale
        self.mask_img = None

    def img_load(self, img_path):
        self.gray_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Initiate mask image with gray image's shape
        self.mask_img = np.zeros(np.shape(self.gray_img))
        # Convert grayscale to RGB
        # self.rgb_img = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2RGB)

    def img_analyze(self):
        height, width = self.gray_img.shape
        print("Image properties:")
        print(f"Height: {height} px")
        print(f"Width: {width} px")

    def _mark_crater_rim(self, longitude, latitude, radius_km):
        # Convert longitude and latitude to pixel coordinates
        center_x = int((longitude - 180) * self.gray_img.shape[1] / 90)
        center_y = int((60 - latitude) * self.gray_img.shape[0] / 60)

        # if 1 <= radius_km < 5:
        #     color = [0, 255, 0]
        # elif 5 <= radius_km < 20:
        #     color = [0, 0, 255]
        # elif radius_km >= 20:
        #     color = [255, 0, 0]
        # else:
        #     color = [0, 0, 0]

        # Place a pixel at the specified coordinates
        self.mask_img[center_y, center_x] = RIM_INTENSITY

        # Draw circumference around center point
        crater_circum_km = 2 * math.pi * radius_km
        steps = int(crater_circum_km/self.scale)
        radius_scaler = 1

        for step in range(steps):
            beta = 2 * math.pi * step / steps
            r_x = radius_scaler * radius_km * math.sin(beta)
            r_y = radius_scaler * radius_km * math.cos(beta)
            latitude_moon_circumference_km = math.sin(math.pi / 2 - math.radians(latitude)) * 2 * math.pi * MEAN_MOON_RADIUS_KM
            gamma_x = math.degrees(r_x * 2 * math.pi / latitude_moon_circumference_km)
            gamma_y = math.degrees(r_y * 2 * math.pi / LONGITUDE_MOON_CIRCUMFERENCE_KM)
            pixel_x = int((longitude + gamma_x - 180) * self.gray_img.shape[1] / 90)
            pixel_y = int((60 - latitude - gamma_y) * self.gray_img.shape[0] / 60)
            # Place a pixel at the specified coordinates
            with contextlib.suppress(IndexError):
                self.mask_img[pixel_y, pixel_x] = RIM_INTENSITY

    def place_craters(self, csv_dir):
        # Read the CSV file with dataset into a Pandas DataFrame
        df = pd.read_csv(csv_dir)
        # Initiate process variables and communicates
        num_rows = df.shape[0]
        print(f"Processing: \"{csv_dir}\" started.")
        process_counter = 0
        print(f"Craters placing: {process_counter}%", end='\r')
        rejected_craters_counter = 0
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Access the second and third columns by their names
            longitude = row['LON_CIRC_IMG']
            latitude = row['LAT_CIRC_IMG']
            # Check if crater center is in bounds
            if 180 < longitude < 270 and 0 < latitude < 60:
                radius = row['DIAM_CIRC_IMG'] / 2
                # minor_axis = row['DIAM_ELLI_MINOR_IMG'] / 2
                # major_axis = row['DIAM_ELLI_MAJOR_IMG'] / 2
                self._mark_crater_rim(longitude, latitude, radius)
            else:
                rejected_craters_counter += 1
            # Update process status
            if int(index/num_rows * 100) > process_counter:
                process_counter = int(index/num_rows * 100)
                print(f"Craters placing: {process_counter}%", end='\r')
        # Process finished display summary
        print("Craters placing 100%")
        print(f"No. rejected craters: {rejected_craters_counter}, it's {rejected_craters_counter/num_rows}%")

    def save_mask(self, output_path):
        if cv2.imwrite(output_path, self.mask_img):
            print("Image saved successfully.")
        else:
            print("Image not saved successfully!")




