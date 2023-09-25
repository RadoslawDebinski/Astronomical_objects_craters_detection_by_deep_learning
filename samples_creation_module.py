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
        cv2.imwrite(output_path, cv2.resize(self.mask_image[y:y + side_size, x:x + side_size],
                                            self.sample_resolution))

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

    def show_distortions_example(self, output_path):
        # sourcery skip: extract-duplicate-method, move-assign-in-block
        # Size of input WAC tile is 27291 x 18194 px it's a 1.5 proportion
        # Size of A4 page with bleed area 3508 x 2480 px
        # So for example width will be 750 and then height 500
        cell_shape = (750, 500)

        # Zooms cords and dimensions N0450
        # first_left_corner = (1300, 16000)
        # second_left_corner = (16000, 18000)
        # zoom_dims = (1950, 1300)
        # Zooms cords and dimensions N2250
        first_left_corner = (0, 3800)
        second_left_corner = (12000, 8300)
        zoom_dims = (3000, 2000)

        # Red border thickness and color
        thickness = 100
        color = (0, 0, 255)  # Red color in BGR format

        # First zoom bordered on image and resize it
        first_red_border = cv2.cvtColor(self.input_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(first_red_border, (first_left_corner[1], first_left_corner[0]),
                      (first_left_corner[1] + zoom_dims[0], first_left_corner[0] + zoom_dims[1]), color, thickness)
        first_red_border = cv2.resize(first_red_border, cell_shape)
        # First zoom clear cropped and convert to BGR
        first_clear = self.input_image[first_left_corner[0]:first_left_corner[0] + zoom_dims[1],
                                       first_left_corner[1]:first_left_corner[1] + zoom_dims[0]]
        first_clear = cv2.cvtColor(first_clear, cv2.COLOR_GRAY2BGR)
        # First mask cropped and converted to BGR
        first_mask = self.mask_image[first_left_corner[0]:first_left_corner[0] + zoom_dims[1],
                                     first_left_corner[1]:first_left_corner[1] + zoom_dims[0]]
        first_mask = cv2.cvtColor(first_mask, cv2.COLOR_GRAY2BGR)
        # First combined = where mask is crater rim make red line where not take clear image
        first_comb = np.where(first_mask == [255, 255, 255], np.full(first_clear.shape, [0, 0, 255]), first_clear)

        # Resize clear, mask and comb
        first_clear = cv2.resize(first_clear.astype(np.uint8), cell_shape)
        first_mask = cv2.resize(first_mask.astype(np.uint8), cell_shape)
        first_comb = cv2.resize(first_comb.astype(np.uint8), cell_shape)

        # Second zoom bordered on image and resize it
        second_red_border = cv2.cvtColor(self.input_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(second_red_border, (second_left_corner[1], second_left_corner[0]),
                      (second_left_corner[1] + zoom_dims[0], second_left_corner[0] + zoom_dims[1]), color,
                      thickness)
        second_red_border = cv2.resize(second_red_border, cell_shape)
        # Second zoom clear cropped and convert to BGR
        second_clear = self.input_image[second_left_corner[0]:second_left_corner[0] + zoom_dims[1],
                                        second_left_corner[1]:second_left_corner[1] + zoom_dims[0]]
        second_clear = cv2.cvtColor(second_clear, cv2.COLOR_GRAY2BGR)
        # Second mask cropped and converted to BGR
        second_mask = self.mask_image[second_left_corner[0]:second_left_corner[0] + zoom_dims[1],
                                      second_left_corner[1]:second_left_corner[1] + zoom_dims[0]]
        second_mask = cv2.cvtColor(second_mask, cv2.COLOR_GRAY2BGR)
        # Second combined = where mask is crater rim make red line where not take clear image
        second_comb = np.where(second_mask == [255, 255, 255], np.full(second_clear.shape, [0, 0, 255]),
                               second_clear)

        # Resize clear, mask and comb
        second_clear = cv2.resize(second_clear.astype(np.uint8), cell_shape)
        second_mask = cv2.resize(second_mask.astype(np.uint8), cell_shape)
        second_comb = cv2.resize(second_comb.astype(np.uint8), cell_shape)

        # Create background
        offset = 20
        combined_image = np.full((cell_shape[1] * 4 + offset * 3, cell_shape[0] * 2 + offset, 3), [255, 255, 255])
        # Paste images on background
        # Element 0, 0
        combined_image[0: cell_shape[1],
                       0: cell_shape[0]] = first_red_border
        # Element 0, 1
        combined_image[0: cell_shape[1],
                       cell_shape[0] + offset: 2 * cell_shape[0] + offset] = second_red_border
        # Element 1, 0
        combined_image[cell_shape[1] + offset: 2 * cell_shape[1] + offset,
                       0: cell_shape[0]] = first_clear
        # Element 1, 1
        combined_image[cell_shape[1] + offset: 2 * cell_shape[1] + offset,
                       cell_shape[0] + offset: 2 * cell_shape[0] + offset] = second_clear
        # Element 2, 0
        combined_image[2 * cell_shape[1] + 2 * offset: 3 * cell_shape[1] + 2 * offset,
                       0: cell_shape[0]] = first_mask
        # Element 2, 1
        combined_image[2 * cell_shape[1] + 2 * offset: 3 * cell_shape[1] + 2 * offset,
                       cell_shape[0] + offset: 2 * cell_shape[0] + offset] = second_mask
        # Element 3, 0
        combined_image[3 * cell_shape[1] + 3 * offset: 4 * cell_shape[1] + 3 * offset,
                       0: cell_shape[0]] = first_comb
        # Element 3, 1
        combined_image[3 * cell_shape[1] + 3 * offset: 4 * cell_shape[1] + 3 * offset,
                       cell_shape[0] + offset: 2 * cell_shape[0] + offset] = second_comb

        # Ensure correct format of combined image
        combined_image = combined_image.astype(np.uint8)
        cv2.imwrite(output_path, combined_image)
        # Display the combined image
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_comparison(self, second_mask_path, output_path):
        # Load second mask image
        second_mask_image = cv2.imread(second_mask_path, cv2.IMREAD_UNCHANGED)
        # Zooms cords and dimensions N2250
        left_corner = (0, 3800)
        zoom_dims = (3000, 2000)
        cell_shape = (750, 500)
        # Create first clear image of area
        first_clear = self.input_image[left_corner[0]:left_corner[0] + zoom_dims[1],
                                       left_corner[1]:left_corner[1] + zoom_dims[0]]
        first_clear = cv2.cvtColor(first_clear, cv2.COLOR_GRAY2BGR)
        # First mask cropped and converted to BGR
        first_mask = self.mask_image[left_corner[0]:left_corner[0] + zoom_dims[1],
                                     left_corner[1]:left_corner[1] + zoom_dims[0]]
        first_mask = cv2.cvtColor(first_mask, cv2.COLOR_GRAY2BGR)
        # First combined = where mask is crater rim make red line where not take clear image
        first_comb = np.where(first_mask == [255, 255, 255], np.full(first_clear.shape, [0, 0, 255]), first_clear)
        # Second mask cropped and converted to BGR
        second_mask = second_mask_image[left_corner[0]:left_corner[0] + zoom_dims[1],
                                        left_corner[1]:left_corner[1] + zoom_dims[0]]
        second_mask = cv2.cvtColor(second_mask, cv2.COLOR_GRAY2BGR)
        # Expand rims at second mask
        # Define the neighborhood size (adjust as needed)
        neighborhood_size = 7
        # Create a kernel for dilation
        kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
        # Dilate the white areas in second_mask
        second_mask = cv2.dilate(second_mask, kernel, iterations=1)
        # First combined = where mask is crater rim make red line where not take clear image
        second_comb = np.where(second_mask == [255, 255, 255], np.full(first_clear.shape, [0, 0, 255]), first_clear)
        # Resize masks combined with clear
        first_comb = cv2.resize(first_comb.astype(np.uint8), cell_shape)
        second_comb = cv2.resize(second_comb.astype(np.uint8), cell_shape)
        # Create background
        offset = 20
        combined_image = np.full((cell_shape[1], cell_shape[0] * 2 + offset, 3), [255, 255, 255])
        # Paste images
        # Element 0, 0
        combined_image[0: cell_shape[1],
                       0: cell_shape[0]] = first_comb
        # Element 0, 1
        combined_image[0: cell_shape[1],
                       cell_shape[0] + offset: 2 * cell_shape[0] + offset] = second_comb

        # Ensure correct format of combined image
        combined_image = combined_image.astype(np.uint8)
        cv2.imwrite(output_path, combined_image)
        # Display the combined image
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_compression_comp(self, output_path):
        # Define edge lengths for areas to compare
        edge_length_km = [50, 100, 150, 200]
        sides_sizes = [int(edge / self.scale_km) for edge in edge_length_km]

        # Define offset between images
        offset = 20
        # Initialize combined image with white bar (upper row)
        combined_masks = np.full((self.sample_resolution[0], offset, 3), [255, 255, 255]).astype(np.uint8)
        # Initialize combined crop with white bar (lower row)
        combined_crops = np.full((self.sample_resolution[0], offset, 3), [255, 255, 255]).astype(np.uint8)

        for side_size in sides_sizes:
            # Crop mask image and resize it
            cropped_mask_image = self.mask_image[0:side_size, 0:side_size]
            resized_mask_image = cv2.resize(cropped_mask_image, self.sample_resolution)
            # Crop from resized mask area of first crop
            new_edge = int(sides_sizes[0] * self.sample_resolution[0] / side_size)
            cropped_cropped_mask = resized_mask_image[0:new_edge, 0:new_edge]
            resized_cropped_mask = cv2.resize(cropped_cropped_mask, self.sample_resolution)
            # Convert mask image and cropped mask to BGR
            resized_mask_image = cv2.cvtColor(resized_mask_image, cv2.COLOR_GRAY2BGR)
            resized_cropped_mask = cv2.cvtColor(resized_cropped_mask, cv2.COLOR_GRAY2BGR)
            # Concatenate the images vertically along their shared edge. Previous concat + new image + white bar
            combined_masks = cv2.hconcat([combined_masks.astype(np.uint8), resized_mask_image.astype(np.uint8),
                                          np.full((self.sample_resolution[0], offset, 3),
                                                  [255, 255, 255]).astype(np.uint8)])
            combined_crops = cv2.hconcat([combined_crops.astype(np.uint8), resized_cropped_mask.astype(np.uint8),
                                          np.full((self.sample_resolution[0], offset, 3),
                                                  [255, 255, 255]).astype(np.uint8)])
        # Combine cropped masks and resided areas with vertical white bar
        combined_example = cv2.vconcat([combined_masks, np.full((offset, combined_masks.shape[1], 3),
                                                                [255, 255, 255]).astype(np.uint8), combined_crops])
        cv2.imwrite(output_path, combined_example)
        cv2.imshow('Combined Image', combined_example)
        cv2.waitKey(0)
        cv2.destroyAllWindows()







