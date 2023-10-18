from dataset_creation.image_annotation_dataset import ImageAnnotationDataset
from dataset_creation.dataset_creation_utils import prep_src_data, create_dataset
from config import CONST_PATH


if __name__ == "__main__":
    # iA = ImageAnnotationDataset(CONST_PATH['trainIN'], CONST_PATH['trainOUT'])
    # input_d, output_d = iA.load_dataset()
    prep_src_data(open_7zip=True, split_catalogue=True, create_masks=True)
    create_dataset(no_samples_train=10, no_samples_valid=10, no_samples_test=10, clear_past=1)
