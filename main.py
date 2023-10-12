import time
import torch
from torchvision import transforms
from image_annotation_dataset import ImageAnnotationDataset

# Dataset paths
DATASET_ROOT = "DatasetRoot"
INPUT_IMAGES = f"{DATASET_ROOT}\\InputImages"
OUTPUT_IMAGES = f"{DATASET_ROOT}\\OutputImages"


if __name__ == "__main__":
    iA = ImageAnnotationDataset(INPUT_IMAGES, OUTPUT_IMAGES)
    input_d, output_d = iA.load_dataset()
    iA.create_metadata()
    iA.create_dataset(1000)
    # iA.show_examples()
