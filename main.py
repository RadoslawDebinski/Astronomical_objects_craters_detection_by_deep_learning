import torch
from torchvision import transforms
from torch.utils.data import Dataset
from dataset_creator import *

# TO DO: Create simple sample creator in creation_module
# create in proper way load_dataset below


class ImageAnnotationDataset(Dataset):
    def __init__(self, input_root_dir, annotation_root_dir, transform=None):
        self.input_root_dir = input_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.transform = transform
        self.input_image_paths, self.annotation_image_paths = self.load_dataset()

    @staticmethod
    def create_metadata():
        # Check if working dirs exists, if not create it
        dir_module()
        # Unpack 7zip to directories
        open_zip_module()
        # Split source catalog for data corresponding to each tile
        source_catalogue_module()
        # Create masks for each tile
        mask_module()

    @staticmethod
    def show_example_dataset():
        example_module()

    @staticmethod
    def create_dataset(no_samples):
        # Check if working dirs exists, if not create it
        dir_module()
        creation_module(no_samples)

    def load_dataset(self):
        return _, _

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        input_image_path = self.input_image_paths[idx]
        annotation_image_path = self.annotation_image_paths[idx]

        input_image = Image.open(input_image_path).convert("RGB")
        annotation_image = Image.open(annotation_image_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            annotation_image = self.transform(annotation_image)

        return input_image, annotation_image


if __name__ == "__main__":
    ImageAnnotationDataset.create_dataset(2)
