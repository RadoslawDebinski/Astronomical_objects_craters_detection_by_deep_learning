from dataset_creation.csv_sorter_module import SourceTypeSeparator
from dataset_creation.mask_module import MaskCreator
from dataset_creation.samples_creation_module import SampleCreator
from config import CONST_PATH, CONST_PATH_CLEAR, INPUT_ZIP_NAME, TILES_NAMES, \
                   CRATERS_CATALOGUE_NAME, \
                   FIRST_COL_ID, CSV_TILES_NAMES, COLS_NAMES_TO_ANALYZE, \
                   SCALE_KM, MEAN_MOON_RADIUS_KM, LONGITUDE_MOON_CIRCUMFERENCE_KM, CRATER_RIM_INTENSITY, \
                   CSV_TILES_KEYS, TILES_BOUNDS, \
                   MIN_CROP_AREA_SIZE_KM, MAX_CROP_AREA_SIZE_KM, SCALE_PX, SAMPLE_RESOLUTION

import py7zr
import os
import time
import shutil


# Preparation of source data for dataset creation process
def prep_src_data(open_7zip=True, split_catalogue=True, create_masks=True, show_examples=False):
    # Check if working dirs exists, if not create it
    print("\nCHECKING DIRECTORIES:\n")
    dir_module()
    if open_7zip:
        # Unpack 7zip to directories
        print("\nUNPACKING DATA:\n")
        open_zip_module()
    if split_catalogue:
        # Split source catalog for data corresponding to each tile
        print("\nSPLITTING CATALOGUE:\n")
        source_catalogue_module()
    if create_masks:
        # Create masks for each tile
        print("\nCREATING MASKS:\n")
        mask_module()
    if show_examples:
        examples_module()


def create_dataset(no_samples_train=0, no_samples_valid=0, no_samples_test=0, clear_past=0):
    # Check if working dirs exists, if not create it
    print("\nCHECKING DIRECTORIES:\n")
    # Clear dataset directories
    if clear_past:
        dir_clear_module()
    dir_module()
    # Start time measurement
    start_time = time.time()
    if no_samples_train:
        print("\nCREATING DATASET TRAINING PART:\n")
        creation_module(no_samples_train, CONST_PATH['trainIN'], CONST_PATH['trainOUT'])
    if no_samples_valid:
        print("\nCREATING DATASET VALIDATION PART:\n")
        creation_module(no_samples_valid, CONST_PATH['validIN'], CONST_PATH['validOUT'])
    if no_samples_test:
        print("\nCREATING DATASET TEST PART:\n")
        creation_module(no_samples_test, CONST_PATH['testIN'], CONST_PATH['testOUT'])
    # End time measurement
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Dataset creator execution time: {execution_time} seconds")


# Check if all directories exists if not make them
def dir_module():
    print(f'Working directory: {os.getcwd()}')
    for path in CONST_PATH.values():
        data_dir = os.path.join(os.getcwd(), path)
        # Check if the directory exists
        if not os.path.exists(data_dir):
            # If it doesn't exist, create it
            os.makedirs(data_dir)
            print(f"Directory '{path}' created.")
        else:
            print(f"Directory '{path}' already exists.")


def dir_clear_module():
    for path in list(CONST_PATH_CLEAR.values()):
        data_dir = os.path.join(os.getcwd(), path)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Directory {path} has been cleared.")
        else:
            print(f"Directory {path} is empty.")


# Load 7zip data
def open_zip_module(input_zip_path=f"{CONST_PATH['tempSRC']}\\{INPUT_ZIP_NAME}"):
    # List of available data names
    data_names = TILES_NAMES.copy() + [CRATERS_CATALOGUE_NAME]
    # Check if the output directory exists; if not, create it
    if not os.path.exists(CONST_PATH['tempSRC']):
        os.makedirs(CONST_PATH['tempSRC'])
    # Iterate through the files in the archive
    for name in data_names:
        with py7zr.SevenZipFile(input_zip_path, mode='r') as archive:
            if name not in archive.getnames():
                raise NotImplementedError(f"{input_zip_path} -> {name} was not opened correctly,"
                                          f"does not exist or contains invalid data")
            print(f"Unzipping {name} file.")
            if name != CRATERS_CATALOGUE_NAME:
                # Extract the WAC tiles to the specified path
                archive.extract(path=CONST_PATH['wacORG'], targets=[name])
                print(f"Unzipped {name} to {CONST_PATH['wacORG']}")
            else:
                # Extract the catalog tiles to the specified path
                archive.extract(path=CONST_PATH['cataORG'], targets=[name])
                print(f"Unzipped {name} to {CONST_PATH['cataORG']}")


# CSV dataset splitting and analysis module handling
def source_catalogue_module(analyze_catalogue=False):
    sTS = SourceTypeSeparator(CONST_PATH['cataORG'], CRATERS_CATALOGUE_NAME, FIRST_COL_ID, CSV_TILES_NAMES,
                              CONST_PATH['cataDIV'])
    sTS.split_craters_by_tile_id()
    if analyze_catalogue:
        sTS.analyze_split_crater_by_tile_id(COLS_NAMES_TO_ANALYZE)


# Images loading, conversion, processing and mask creation module handling
def mask_module():
    iMGA = MaskCreator(SCALE_KM, MEAN_MOON_RADIUS_KM, LONGITUDE_MOON_CIRCUMFERENCE_KM, CRATER_RIM_INTENSITY)
    # Iteration throw created CSV files with rejection of polar images: P900S, P900N
    for index, tile in enumerate(TILES_NAMES):
        iMGA.img_load(os.path.join(CONST_PATH['wacORG'], tile))
        # iMGA.img_analyze()
        key = CSV_TILES_KEYS[index]
        iMGA.place_craters(f"{CONST_PATH['cataDIV']}\\{CSV_TILES_NAMES[key]}.csv", TILES_BOUNDS[index])
        iMGA.save_mask(f"{CONST_PATH['wacMASK']}\\MASK_{CSV_TILES_NAMES[key]}.jpg")


def examples_module():
    scale_px = 1 / SCALE_KM
    # Iteration through created CSV files with rejection of polar images: P900S, P900N
    for index, tile in enumerate(TILES_NAMES):
        key = CSV_TILES_KEYS[index]
        # Feeding Sample creator with parameters
        sC = SampleCreator(MIN_CROP_AREA_SIZE_KM * scale_px, MAX_CROP_AREA_SIZE_KM * scale_px, SAMPLE_RESOLUTION,
                           SCALE_KM, os.path.join(CONST_PATH['wacORG'], tile),
                           f"{CONST_PATH['wacMASK']}\\MASK_{CSV_TILES_NAMES[key]}.jpg")
        sC.show_random_samples()


def creation_module(no_samples, input_path, output_path):
    # How many samples create per tile
    no_samples_per_tile = int(no_samples / len(TILES_NAMES))
    # Iteration through created CSV files with rejection of polar images: P900S, P900N
    for index, tile in enumerate(TILES_NAMES):
        # Info which tile is during processing
        print(f"Processing tile no.{index + 1}: {tile}")
        key = CSV_TILES_KEYS[index]
        # Create base name for sample
        file_name = "0" * len(str(no_samples_per_tile))
        # Feeding Sample creator with parameters
        sC = SampleCreator(MIN_CROP_AREA_SIZE_KM * SCALE_PX, MAX_CROP_AREA_SIZE_KM * SCALE_PX, SAMPLE_RESOLUTION,
                           SCALE_KM, os.path.join(CONST_PATH['wacORG'], tile),
                           f"{CONST_PATH['wacMASK']}\\MASK_{CSV_TILES_NAMES[key]}.jpg")
        # Create file names
        file_names = [
            f"{index + 1}_{str(int(file_name) + i).zfill(len(file_name))}"
            for i in range(no_samples_per_tile)
        ]

        # Define a lambda function to create samples
        for name in file_names:
            sC.make_sample(f"{input_path}\\{name}.jpg", f"{output_path}\\{name}.jpg")
        print(f"{no_samples_per_tile} samples created")
