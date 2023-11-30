from dataset_creation.dataset_creation_utils import prep_src_data, create_dataset
from demo_utils.train_demo_utils import train_model_const, check_model_const, test_model_const
from transfer_learning.mars_processing import MarsSamples
from settings import CONST_PATH, INPUT_ZIP_NAME, EXAMPLE_AU_NET_F32_MODEL, EXAMPLE_INPUTS, EXAMPLE_MASKS


def check_example_model_option():
    """
    Check pretrained model F=32 and show results on example samples
    """
    model_path = f"{CONST_PATH['example']}/{EXAMPLE_AU_NET_F32_MODEL}"
    input_images_path = [f"{CONST_PATH['example']}/{example_input}" for example_input in EXAMPLE_INPUTS]
    output_images_path = [f"{CONST_PATH['example']}/{example_output}" for example_output in EXAMPLE_MASKS]
    for i in range(len(input_images_path)):
        check_model_const(model_path, input_images_path[i], output_images_path[i])

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
    input("Press Enter to start training process...")
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
    Test Attention U-Net model on Mars test dataset with constants from 'settings.py'
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
    model_path_choice = input(f"Enter the path to Attention U-Net model trained in this project: ")
    input_image_choice = input(f"Enter the path to original sample (e.g. from test set): ")
    output_image_choice = input(f"Enter the path to expected mask (e.g. from test set): ")

    check_model_const(model_path_choice, input_image_choice, output_image_choice)


if __name__ == "__main__":
    # === DEBUG ===
    # train_model_const()
    # exit()
    # =============

    print("Craters detection on Moon using Attention U-Net along with tests on Mars data.")
    print("More information (e.g. theory) you can find in engineering thesis: "
          "'Astronomical objects craters detection by deep learning'.")
    print("Authors: Radosław Dębiński, Tomash Mikulevich.")
    while True:
        print("=" * 60)
        print("Please choose an operation:")
        print("1. [Examples] Show results of pretrained model (32 filters) on example samples of Moon and Mars.")
        print("2. Generate Moon dataset: train, validation and test set.")
        print("3. Create Mars dataset: test set.")
        print("4. Train demo model from scratch.")
        print("5. Test model with Moon test set.")
        print("6. Test model with Mars test set.")
        print("7. Check model with one sample and plot comparison.")
        print("8. Exit")

        choice = input("Your choice: ")
        if choice == "1":
            check_example_model_option()
        elif choice == "2":
            create_moon_dataset_option()
        elif choice == "3":
            create_mars_dataset_option()
        elif choice == "4":
            train_model_option()
        elif choice == "5":
            test_model_moon_option()
        elif choice == "6":
            test_model_mars_option()
        elif choice == "7":
            check_model_option()
        elif choice == "8":
            print("Exit...")
            break
        else:
            print("Unknown option. Try again...")
