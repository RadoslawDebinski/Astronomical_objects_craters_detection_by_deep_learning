import contextlib
import cv2
import numpy as np
import pandas as pd
import math
import random


class DatasetCreator:
    def __init__(self, min_side_size_px, max_side_size_px, sample_resolution, scale_km):
        self.min_side_size_px = min_side_size_px
        self.max_side_size_px = max_side_size_px
        self.sample_resolution = sample_resolution
        self.scale_km = scale_km

    # Rotation - for eventually further usage
    @staticmethod
    def _rotate_image(image, angle):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the center of the image
        center = (width // 2, height // 2)

        # Define the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation to the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        return rotated_image

    def run_pipeline(self):
        pass

    def show_sample(self, image_path, mask_path):
        # Load source image and mask
        input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # Ensure same size of them
        if input_image.shape == mask_image.shape:
            # Define the range for random side size (min_side_size_px to max_side_size_px)
            side_size = random.randint(self.min_side_size_px, self.max_side_size_px)

            # Get the dimensions of the input image
            image_height, image_width = input_image.shape[:2]

            # Generate random (x, y) coordinates for the top-left corner of the square
            x = random.randint(0, image_width - side_size)
            y = random.randint(0, image_height - side_size)

            # Calculate the bottom-right corner of the square
            x2 = x + side_size
            y2 = y + side_size

            # Generate a random angle between -45 and 45 degrees (adjust as needed)
            angle = random.randint(0, 359)

            # Crop the square region from the input image
            # cropped_input_image = self._rotate_image(input_image[y:y2, x:x2], angle)
            # cropped_mask_image = self._rotate_image(mask_image[y:y2, x:x2], angle)
            cropped_input_image = input_image[y:y2, x:x2]
            cropped_mask_image = mask_image[y:y2, x:x2]

            resized_input_image = cv2.resize(cropped_input_image, self.sample_resolution)
            resized_mask_image = cv2.resize(cropped_mask_image, self.sample_resolution)

            # Concatenate the images vertically along their shared edge
            combined_image = cv2.hconcat([resized_input_image, resized_mask_image])

            # Info for user
            edge_length_km = int(side_size * self.scale_km)
            print(f'Presented area size is: {edge_length_km}x{edge_length_km}km')

            # Display the combined image
            cv2.imshow('Combined Image', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

