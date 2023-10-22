from dataset_creation.dataset_creation_utils import prep_src_data, create_dataset
from train_demo_utils import train_model_const, check_model_const


if __name__ == "__main__":
    # prep_src_data(open_7zip=True, split_catalogue=True, create_masks=True)
    # create_dataset(no_samples_train=10, no_samples_valid=10, no_samples_test=10, clear_past=True)
    train_model_const()
