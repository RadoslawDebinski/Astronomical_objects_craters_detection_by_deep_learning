import time
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from dataset_creation_utils import *
from PIL import Image


class ImageAnnotationDataset(Dataset):
    def __init__(self, input_root_dir, annotation_root_dir, transform=None):
        # Check if working dirs exists, if not create it
        print("\nCHECKING DIRECTORIES:\n")
        dir_module()
        self.input_root_dir = input_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.transform = transform
        self.input_image_paths, self.annotation_image_paths = self.load_dataset()

    @staticmethod
    def create_metadata():
        # Check if working dirs exists, if not create it
        print("\nCHECKING DIRECTORIES:\n")
        dir_module()
        # Unpack 7zip to directories
        print("\nUNPACKING DATA:\n")
        # open_zip_module()
        # Split source catalog for data corresponding to each tile
        print("\nSPLITTING CATALOGUE:\n")
        # source_catalogue_module()
        # Create masks for each tile
        print("\nCREATING MASKS:\n")
        mask_module()

    @staticmethod
    def show_examples():
        examples_module()

    @staticmethod
    def create_dataset(no_samples):
        # Check if working dirs exists, if not create it
        print("\nCHECKING DIRECTORIES:\n")
        dir_module()
        start_time = time.time()
        # Create dataset
        print("\nCREATING DATASET:\n")
        creation_module(no_samples)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Dataset creator execution time: {execution_time} seconds")

    def load_dataset(self):
        input_files = sorted(os.listdir(self.input_root_dir))
        input_files = [os.path.join(self.input_root_dir, file) for file in input_files]

        annotation_files = sorted(os.listdir(self.annotation_root_dir))
        annotation_files = [os.path.join(self.annotation_root_dir, file) for file in annotation_files]
        return input_files, annotation_files

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        input_image_path = self.input_image_paths[idx]
        annotation_image_path = self.annotation_image_paths[idx]

        input_image = Image.open(input_image_path)
        annotation_image = Image.open(annotation_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            annotation_image = self.transform(annotation_image)

        return input_image, annotation_image

