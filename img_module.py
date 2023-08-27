import cv2
import pandas as pd
import matplotlib.pyplot as plt

class ImgAnalyzer:
    def __init__(self):
        self.gray_img = None
        self.rgb_img = None

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
        x = int((longitude - 180) * self.rgb_img.shape[1] / 90)
        y = int((60 - latitude) * self.rgb_img.shape[0] / 60)
        # Convert radius from kilometers to pixels using the scale of 100m/px
        radius_px = int(radius_km * 10)  # 100m = 1px, so 1km = 10px
        if 1.5 <= radius_km < 5:
            color = [0, 255, 0]
        elif 5 <= radius_km < 20:
            color = [0, 0, 255]
        elif radius_km >= 20:
            color = [255, 0, 0]
        else:
            color = [0, 0, 0]

        # Place a pixel at the specified coordinates
        self.rgb_img[y, x] = color

        # Draw a blue circle around the pixel
        cv2.circle(self.rgb_img, (x, y), radius_px, tuple(color), 2)  # Blue: [R, G, B]

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




