from dataset_creation.image_annotation_dataset import ImageAnnotationDataset
from dataset_creation.dataset_creation_utils import prep_src_data, create_dataset
from config import CONST_PATH


if __name__ == "__main__":
    # iA = ImageAnnotationDataset(CONST_PATH['trainIN'], CONST_PATH['trainOUT'])
    # input_d, output_d = iA.load_dataset()
    prep_src_data(open_7zip=False, split_catalogue=False, create_masks=False)
    create_dataset(no_samples_train=0, no_samples_valid=0, no_samples_test=0, clear_past=1)
