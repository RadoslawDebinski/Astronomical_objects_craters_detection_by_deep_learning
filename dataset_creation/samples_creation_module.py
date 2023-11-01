import cv2
import numpy as np
import random
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from settings import RESOLUTION


class SampleCreator:
    def __init__(self, min_side_size_px, max_side_size_px, sample_resolution, scale_km, image_path, mask_path):
        self.min_side_size_px = min_side_size_px
        self.max_side_size_px = max_side_size_px
        self.sample_resolution = sample_resolution
        self.scale_km = scale_km
        self.input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        self.image_height, self.image_width = self.input_image.shape[:2]

    @staticmethod
    def _rotate_image(image, angle):
        """
        Rotation of images - for eventually further usage
        """
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the center of the image
        center = (width // 2, height // 2)

        # Define the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def make_sample(self, input_path, output_path):
        """
        Create the right sample for dataset
        """
        # Define the range for random side size (from min_side_size_px to max_side_size_px)
        side_size = random.randint(self.min_side_size_px, self.max_side_size_px)
        # Generate random (x, y) coordinates for the top-left corner of the square
        x = random.randint(0, self.image_width - side_size)
        y = random.randint(0, self.image_height - side_size)

        # TODO Adding ccrs.Orthographic projection to output

        # Cropping, resizing and saving images
        cv2.imwrite(input_path, cv2.resize(self.input_image[y:y + side_size, x:x + side_size],
                                           self.sample_resolution))
        resized_mask = cv2.resize(self.mask_image[y:y + side_size, x:x + side_size],
                                  self.sample_resolution)
        resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)[1]

        cv2.imwrite(output_path, resized_mask)

    def show_random_samples(self, bounds):
        """
        This feature is intended to provide sample data of a future dataset
        """
        # Define the range for random side size (min_side_size_px to max_side_size_px)
        side_size = random.randint(self.min_side_size_px, self.max_side_size_px)

        # Get the dimensions of the input image
        image_height, image_width = self.input_image.shape[:2]

        # Generate random (x, y) coordinates for the top-left corner of the square
        x = random.randint(0, image_width - side_size)
        y = random.randint(0, image_height - side_size)

        # Calculate the bottom-right corner of the square
        x2 = x + side_size
        y2 = y + side_size

        # Crop images
        cropped_input_image = self.input_image[y:y2, x:x2]
        cropped_mask_image = self.mask_image[y:y2, x:x2]

        # Get longitude and latitude of center
        long_limit, lat_limit = bounds[2], bounds[1]

        center_px_x = (x + x2) / 2
        center_long = center_px_x * 90 / self.input_image.shape[1] + long_limit
        center_px_y = (y + y2) / 2
        center_lat = lat_limit - center_px_y * 60 / self.input_image.shape[0]

        # projection = ccrs.Orthographic(
        #     central_longitude=0.0,  # Centered at the Moon's prime meridian
        #     central_latitude=0.0,  # Centered at the Moon's equator
        #     globe=ccrs.Globe(semimajor_axis=1738100.0, semiminor_axis=1737400.0)
        #     # Moon's semimajor and semiminor axes
        # )

        projection = ccrs.Geostationary(central_longitude=0.0, satellite_height=1737400.0, globe=None)

        # Define extent in degrees
        deg_per_px = 1 / RESOLUTION
        extent = [center_long - (side_size / 2) * deg_per_px, center_long + (side_size / 2) * deg_per_px,
                  center_lat - (side_size / 2) * deg_per_px, center_lat + (side_size / 2) * deg_per_px]

        # Create a figure and axis with the orthographic projection
        fig, ax = plt.subplots(subplot_kw={'projection': projection})

        # Plot the Moon image on the orthographic projection for input
        ax.imshow(cropped_input_image, extent=extent, transform=ccrs.PlateCarree(), origin='upper')
        # Remove the axes
        ax.set_axis_off()

        # Create a FigureCanvasAgg instance to render the figure
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        transformed_input_image = np.array(fig.canvas.renderer.buffer_rgba())
        # Plot the Moon image on the orthographic projection for output
        ax.imshow(cropped_mask_image, extent=extent, transform=ccrs.PlateCarree(), origin='upper')
        # Remove the axes
        ax.set_axis_off()

        # Create a FigureCanvasAgg instance to render the figure
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        transformed_mask_image = np.array(fig.canvas.renderer.buffer_rgba())

        combined_image_pure = cv2.hconcat([cv2.resize(cropped_input_image, self.sample_resolution),
                                           cv2.resize(cropped_mask_image, self.sample_resolution)])

        combined_image_trans = cv2.hconcat([cv2.resize(transformed_input_image, self.sample_resolution),
                                            cv2.resize(transformed_mask_image, self.sample_resolution)])

        # Display the combined image
        cv2.imshow('Combined Image',  cv2.vconcat([combined_image_pure, combined_image_trans]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

