import os
from torch.utils.data import Dataset
from dataset_creation.dataset_creation_utils import dir_module, creation_module
from PIL import Image


class ImageAnnotationDataset(Dataset):
    """
    This class is extended version of standard approach to custom dataset loader in torch library
    """
    def __init__(self, input_root_dir, annotation_root_dir, transform=None):
        # Check if working dirs exists, if not create it
        print("\nCHECKING DIRECTORIES:\n")
        dir_module()
        self.input_root_dir = input_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.transform = transform
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

        input_image = Image.open(input_image_path)
        annotation_image = Image.open(annotation_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            annotation_image = self.transform(annotation_image)

        return input_image, annotation_image

