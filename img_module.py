import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
import os


MOON_CIRCUMFERENCE_KM = 10920
SCALE_KM = 0.1


class ImgAnalyzer:
    def __init__(self, scale, resolution):
        self.gray_img = None
        self.rgb_img = None
        self.scale = scale
        self.resolution = resolution
        self.ellipse_counter = 0

    def img_load_convert(self, img_path):
        self.gray_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Convert grayscale to RGB
        self.rgb_img = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2RGB)

    def img_analyze(self, img):
        height, width, channels = img.shape
        print("Image properties:")
        print(f"Height: {height} px")
        print(f"Width: {width} px")
        print(f"No. channels: {channels}")
        print(f"No. funny ellipses {self.ellipse_counter}")
        self._img_display()

    def _img_display(self):
        plt.imshow(self.rgb_img)
        plt.show()

    def _mark_crater(self, longitude, latitude, radius_km, major_axis, minor_axis):
        # Convert longitude and latitude to pixel coordinates
        center_x = int((longitude - 180) * self.rgb_img.shape[1] / 90)
        center_y = int((60 - latitude) * self.rgb_img.shape[0] / 60)

        if 1.5 <= radius_km < 5:
            color = [0, 255, 0]
        elif 5 <= radius_km < 20:
            color = [0, 0, 255]
        elif radius_km >= 20:
            color = [255, 0, 0]
        else:
            color = [0, 0, 0]

        # Place a pixel at the specified coordinates
        self.rgb_img[center_y, center_x] = color

        # Draw circumference around center point
        crater_circum_km = 2 * math.pi * radius_km
        steps = int(crater_circum_km/SCALE_KM)

        for step in range(steps):
            beta = 2 * math.pi * step / steps
            r_x = radius_km * math.sin(beta)
            r_y = radius_km * math.cos(beta)
            gamma_x = r_x * 2 * math.pi / MOON_CIRCUMFERENCE_KM
            gamma_y = r_y * 2 * math.pi / MOON_CIRCUMFERENCE_KM
            pixel_x = int((longitude + gamma_x - 180) * self.rgb_img.shape[1] / 90)
            pixel_y = int((60 - latitude - gamma_y) * self.rgb_img.shape[0] / 60)
            # Place a pixel at the specified coordinates
            try:
                self.rgb_img[pixel_y, pixel_x] = color
                self.rgb_img[pixel_y, pixel_x+1] = color
                self.rgb_img[pixel_y, pixel_x-1] = color
                self.rgb_img[pixel_y+1, pixel_x] = color
                self.rgb_img[pixel_y-1, pixel_x] = color
            except IndexError:
                print("IndexError detected")



    def place_craters_centers(self, csv_dir):
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(csv_dir)

        num_rows = df.shape[0]

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Access the second and third columns by their names
            longitude = row['LON_CIRC_IMG']
            latitude = row['LAT_CIRC_IMG']

            if 180 < longitude < 270 and 0 < latitude < 60:
                radius = row['DIAM_CIRC_IMG'] / 2
                minor_axis = row['DIAM_ELLI_MINOR_IMG'] / 2
                major_axis = row['DIAM_ELLI_MAJOR_IMG'] / 2
                self._mark_crater(longitude, latitude, radius, major_axis, minor_axis)
            print(f"Craters placing: {round(index/num_rows * 100, 3)}%")

    @staticmethod
    def save_image(output_path, image):
        cv2.imwrite(output_path, image)
        print("Image saved successfully.")




