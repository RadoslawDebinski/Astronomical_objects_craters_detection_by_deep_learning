import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt

class ImgAnalyzer:
    def __init__(self, scale, resolution):
        self.gray_img = None
        self.rgb_img = None
        self.scale = scale
        self.resolution = resolution

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
        self._img_display()

    def _img_display(self):
        plt.imshow(self.rgb_img)
        plt.show()

    def _mark_crater(self, longitude, latitude, radius_km):
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
        circle = self._mark_circle(center_x, center_y, radius_km)

        for point in circle:
            if 0 < point[0] < self.rgb_img.shape[0] and 0 < point[1] < self.rgb_img.shape[1]:
                self.rgb_img[int(point[0]), int(point[1])] = color

    def _mark_circle(self, center_x, center_y, radius_km):
        # Convert radius from meters to degrees
        radius_m = 1000 * radius_km
        radius_deg = radius_m / self.scale

        # Calculate the number of steps in longitude and latitude
        # You can adjust the step size (smaller for higher precision)
        step_size = 1 / self.resolution
        longitude_steps = math.ceil(360 * radius_deg)
        latitude_steps = math.ceil(180 * radius_deg)

        # Fixed Value !!! Will be changed !!!
        points_per_circle = 120  # number of points per circle
        # Calculate angle between points
        angle_step = 360 / points_per_circle
        # List to store calculated points
        points = []

        # Iterate through longitude and latitude within the specified radius
        for i in range(points_per_circle):
            angle_rad = math.radians(i * angle_step)
            x = center_x + radius_deg * math.cos(angle_rad) * self.resolution
            y = center_y - radius_deg * math.sin(angle_rad) * self.resolution

            # Convert pixel coordinates to longitude and latitude
            longitude = (x / self.resolution) - 180
            latitude = 90 - (y / self.resolution)

            points.append((latitude, longitude))

        return points

    def place_craters_centers(self, csv_dir):
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(csv_dir)

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Access the second and third columns by their names
            longitude = row['LON_CIRC_IMG']  # Replace 'Column2' with the actual column name
            latitude = row['LAT_CIRC_IMG']  # Replace 'Column3' with the actual column name
            radius = row['DIAM_CIRC_IMG'] / 2

            if 180 < longitude < 270 and 0 < latitude < 60:
                self._mark_crater(longitude, latitude, radius)

    @staticmethod
    def save_image(output_path, image):
        cv2.imwrite(output_path, image)
        print("Image saved successfully.")




