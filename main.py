from dataset_creation.dataset_creation_utils import prep_src_data, create_dataset
from demo_utils.train_demo_utils import train_model_const, check_model_const, test_model_const
from transfer_learning.mars_processing import MarsSamples
from settings import CONST_PATH, INPUT_ZIP_NAME

def display_info_option():
    """
    Display information about the entire project.
    TODO: add abstract from engineering thesis
    """
    print("<<< Astronomical objects craters detection by deep learning >>>")

def create_moon_dataset_option():
    """
    Generate Moon dataset using provided info from keyboard and constants from 'settings.py'
    """
    have_zip = input(f"Do you have file {INPUT_ZIP_NAME} located in {CONST_PATH['source']}? [y/n]: ")
    if have_zip != "y":
        print("You can't generate dataset without mentioned .zip. It contains source images and catalogs.")
        return

    prep_src_option = input(f"Do you want to prepare source data? "
                            f"E.g. if you download {INPUT_ZIP_NAME} and do nothing with it "
                            f"or you want to recreate masks "
                            f"[y/n]: ")
    if prep_src_option == "y":
        open_zip_choice = input(f"Open {INPUT_ZIP_NAME}? [y/n]: ")
        split_catalogue_choice = input(f"Split catalogs? [y/n]: ")
        create_mask_choice = input(f"Create masks? [y/n]: ")

        open_zip_choice = True if open_zip_choice == "y" else False
        split_catalogue_choice = True if split_catalogue_choice == "y" else False
        create_mask_choice = True if create_mask_choice == "y" else False

        prep_src_data(open_7zip=open_zip_choice, split_catalogue=split_catalogue_choice, create_masks=create_mask_choice)

    create_dataset_option = input(f"Continue generating dataset? "
                                  f"Make sure you have prepared source data "
                                  f"[y/n]: ")
    if create_dataset_option == "y":
        clear_past_choice = input(f"Do you want to delete previous samples? [y/n]: ")
        train_no_choice = int(input(f"Enter how many train samples you want to generate: "))
        valid_no_choice = int(input(f"Enter how many validation samples you want to generate: "))
        test_no_choice = int(input(f"Enter how many test samples you want to generate: "))

        clear_past_choice = True if clear_past_choice == "y" else False

        create_dataset(no_samples_train=train_no_choice, no_samples_valid=valid_no_choice, no_samples_test=test_no_choice,
                       clear_past=clear_past_choice)

def create_mars_dataset_option():
    """
    Generate Mars dataset using provided info from keyboard and constants from 'settings.py'
    """
    print("Warning: make sure you have stable Internet connection.")
    mars_samples_choice = int(input(f"Enter how many test samples you want to generate: "))
    ms = MarsSamples(no_samples=mars_samples_choice)
    ms.create_dataset()

def train_model_option():
    """
    Train Attention U-Net model from zero using constants from 'settings.py'
    """
    print("Warning: make sure you have generated Moon train and validating dataset "
          "and have parameters of the network in settings.py.")
    print("Start training process...")
    train_model_const()

def test_model_moon_option():
    """
    Test Attention U-Net model on Moon test dataset with constants from 'settings.py'
    """
    print("Warning: make sure you have generated Moon test dataset "
          "and have right parameters of the network in settings.py.")
    model_path_choice = input(f"Enter the path to Attention U-Net trained in this project: ")

    print("Start testing process...")
    test_model_const(model_path_choice, "moon")

def test_model_mars_option():
    """
    Test Attention U-Net model on Mars test dataset (transfer learning) with constants from 'settings.py'
    """
    print("Warning: make sure you have generated Mars test dataset "
          "and have right parameters of the network in settings.py.")
    model_path_choice = input(f"Enter the path to Attention U-Net trained in this project: ")

    print("Start testing process...")
    test_model_const(model_path_choice, "mars")

def check_model_option():
    """
    Train Attention U-Net model using constants from 'settings.py'
    """
    model_path_choice = input(f"Enter the path to Attention U-Net trained in this project: ")
    input_image_choice = input(f"Enter the path to original sample 256x256 (e.g. from test set): ")
    output_image_choice = input(f"Enter the path to expected mask 256x256 (e.g. from test set): ")

    check_model_const(model_path_choice, input_image_choice, output_image_choice)


if __name__ == "__main__":
    # === DEBUG ===
    # train_model_const()
    # exit()
    # =============

    print("Craters detection on Moon using Attention U-Net along with transfer learning on Mars data.")
    print("Authors: Radosław Dębiński, Tomash Mikulevich.")
    while True:
        print("=" * 60)
        print("Please choose an operation:")
        print("0. Display additional information")
        print("1. Generate Moon dataset: train, validation and test set.")
        print("2. Create Mars dataset: test set for transfer learning.")
        print("3. Train demo model from scratch.")
        print("4. Test your model with Moon test set.")
        print("5. Test your model with Mars test set.")
        print("6. Check your model with one sample and plot comparison.")
        print("7. [Exit]")

        choice = input("Your choice: ")
        if choice == "0":
            display_info_option()
        elif choice == "1":
            create_moon_dataset_option()
        elif choice == "2":
            create_mars_dataset_option()
        elif choice == "3":
            train_model_option()
        elif choice == "4":
            test_model_moon_option()
        elif choice == "5":
            test_model_mars_option()
        elif choice == "6":
            check_model_option()
        elif choice == "7":
            print("Exit...")
            break
        else:
            print("Unknown option. Try again...")
