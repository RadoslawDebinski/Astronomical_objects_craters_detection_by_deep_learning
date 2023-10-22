from dataset_creation.dataset_creation_utils import dir_module

import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image


class ImageAnnotationDataset(Dataset):
    """
    This class is extended version of standard approach to custom dataset loader in torch library
    """
    def __init__(self, input_root_dir, annotation_root_dir, transforms_input=None, transforms_annotation=None):
        print("\nCHECKING DIRECTORIES:\n")
        dir_module()
        self.input_root_dir = input_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.transforms_input = Compose([ToTensor()] if transforms_input is None else transforms_input)
        self.transforms_annotation = Compose([ToTensor()] if transforms_annotation is None else transforms_annotation)
        self.input_image_paths, self.annotation_image_paths = self.load_dataset()

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

        input_image = self.transforms_input(Image.open(input_image_path))
        annotation_image = self.transforms_annotation(Image.open(annotation_image_path))

        return input_image, annotation_image
