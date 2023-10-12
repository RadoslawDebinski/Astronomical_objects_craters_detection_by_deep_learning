import contextlib
import cv2
import numpy as np
import pandas as pd
import math
import random
import py7zr


class SampleCreator:
    def __init__(self, min_side_size_px, max_side_size_px, sample_resolution, scale_km, image_path, mask_path):
        self.min_side_size_px = min_side_size_px
        self.max_side_size_px = max_side_size_px
        self.sample_resolution = sample_resolution
        self.scale_km = scale_km
        self.input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        self.image_height, self.image_width = self.input_image.shape[:2]

    # Rotation - for eventually further usage
    @staticmethod
    def _rotate_image(image, angle):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the center of the image
        center = (width // 2, height // 2)

        # Define the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def make_sample(self, input_path, output_path):
        # Define the range for random side size (min_side_size_px to max_side_size_px)
        side_size = random.randint(self.min_side_size_px, self.max_side_size_px)
        # Generate random (x, y) coordinates for the top-left corner of the square
        x = random.randint(0, self.image_width - side_size)
        y = random.randint(0, self.image_height - side_size)
        # Cropping, resizing and saving images
        cv2.imwrite(input_path, cv2.resize(self.input_image[y:y + side_size, x:x + side_size],
                                           self.sample_resolution))
        resized_mask = cv2.resize(self.mask_image[y:y + side_size, x:x + side_size],
                                  self.sample_resolution)
        resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)[1]
        # resized_mask = np.where(resized_mask > 127, 1, 0)
        cv2.imwrite(output_path, resized_mask)

    def show_random_samples(self):
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

        cropped_input_image = self.input_image[y:y2, x:x2]
        cropped_mask_image = self.mask_image[y:y2, x:x2]

        resized_input_image = cv2.resize(cropped_input_image, self.sample_resolution)
        resized_mask_image = cv2.resize(cropped_mask_image, self.sample_resolution)

        resized_mask_image = cv2.threshold(resized_mask_image, 127, 255, cv2.THRESH_BINARY)[1]

        # Concatenate the images vertically along their shared edge
        combined_image = cv2.hconcat([resized_input_image, resized_mask_image])

        # Info for user
        edge_length_km = int(side_size * self.scale_km)
        print(f'Presented area cords are x from: {x}px to: {x2}px, y from: {y}px to: {y2}px')
        print(f'Presented area size is: {edge_length_km}x{edge_length_km}km')

        # Display the combined image
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()







