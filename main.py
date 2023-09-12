from csv_sorter_module import SourceTypeSeparator
from mask_module import MaskCreator
from dataset_creation_module import DatasetCreator
import re
import py7zr
import os

# TODO: Remade mask module to create mask files for all tiles
# TODO: Make run_pipeline in dataset_creation_module
# TODO: Test if dataset creator works

# Input and temp directories / names
INPUT_ZIP_PATH = "InputData.7z"
INPUT_DATA_PATH = "InputData"
CRATERS_CATALOGUE_NAME = "data_lunar_crater_database_robbins_2018.csv"
TEMP_CRATERS_BY_TILE_DIR = "TempData"

# Constants for proper CSV and tales processing
CSV_TILES_NAMES = {
    "00-\d-\d{6}": "WAC_GLOBAL_P900S0000_100M",
    "01-\d-\d{6}": "WAC_GLOBAL_P900N0000_100M",
    "02-\d-\d{6}": "WAC_GLOBAL_E300N2250_100M",
    "03-\d-\d{6}": "WAC_GLOBAL_E300S2250_100M",
    "04-\d-\d{6}": "WAC_GLOBAL_E300N3150_100M",
    "05-\d-\d{6}": "WAC_GLOBAL_E300S3150_100M",
    "06-\d-\d{6}": "WAC_GLOBAL_E300N0450_100M",
    "07-\d-\d{6}": "WAC_GLOBAL_E300S0450_100M",
    "08-\d-\d{6}": "WAC_GLOBAL_E300N1350_100M",
    "09-\d-\d{6}": "WAC_GLOBAL_E300S1350_100M",
}
CSV_TILES_KEYS = list(CSV_TILES_NAMES.keys())
TILES_NAMES = [f'{name}.tif' for name in list(CSV_TILES_NAMES.values())]
FIRST_COL_ID = "CRATER_ID"
COLS_NAMES_TO_ANALYZE = ["LAT_CIRC_IMG", "LON_CIRC_IMG", "LAT_ELLI_IMG", "LON_ELLI_IMG"]

# WAC tiles images constants
SCALE_KM = 0.1  # kilometers per pixel
RESOLUTION = 303.23  # pixels per degree

# End dataset properties
MIN_CROP_AREA_SIZE_KM = 500
MAX_CROP_AREA_SIZE_KM = 1000
SAMPLE_RESOLUTION = (512, 512)

# Moon constants
MEAN_MOON_RADIUS_KM = 1737.05
LONGITUDE_MOON_CIRCUMFERENCE_KM = 10907

# Other constants
CRATER_RIM_INTENSITY = 255


# Load 7zip data
def open_zip_module():
    # List of available data names
    data_names = TILES_NAMES.copy() + [CRATERS_CATALOGUE_NAME]
    # Check if archive contains valid names without polar images: P900S, P900N
    print(f"Opening {INPUT_DATA_PATH} directory.")
    # Check if the output directory exists; if not, create it
    if not os.path.exists(INPUT_DATA_PATH):
        os.makedirs(INPUT_DATA_PATH)
    # Iterate through the files in the archive ignore first two pools: P900S0000 and P900N0000
    for name in data_names[2:]:
        # Open 7z
        print(f"Opening {INPUT_ZIP_PATH} file.")
        with py7zr.SevenZipFile(INPUT_ZIP_PATH, mode='r') as archive:
            if name not in archive.getnames():
                raise NotImplementedError(f"{INPUT_ZIP_PATH} was not opened correctly, does not exist or contains"
                                          f"invalid data")
            print(f"Unzipping {name} file.")
            # Extract the file to the specified path
            archive.extract(path=INPUT_DATA_PATH, targets=[name])


# CSV dataset splitting and analysis module handling
def source_catalogue_module():
    sTS = SourceTypeSeparator(INPUT_DATA_PATH, CRATERS_CATALOGUE_NAME, FIRST_COL_ID, CSV_TILES_NAMES,
                              TEMP_CRATERS_BY_TILE_DIR)
    sTS.split_craters_by_tile_id()
    # sTS.analyze_split_crater_by_tile_id(COLS_NAMES_TO_ANALYZE)


# Images loading, conversion, processing and mask creation module handling
def mask_module():
    iMGA = MaskCreator(SCALE_KM, MEAN_MOON_RADIUS_KM, LONGITUDE_MOON_CIRCUMFERENCE_KM, CRATER_RIM_INTENSITY)
    for index, tile in enumerate(TILES_NAMES[2:], start=2):
        iMGA.img_load(load_zip_module(), tile)
        iMGA.img_analyze()
        key = CSV_TILES_KEYS[index]
        iMGA.place_craters(f"{TEMP_CRATERS_BY_TILE_DIR}\\{CSV_TILES_NAMES[key]}.csv")
        iMGA.save_mask(f"{TEMP_CRATERS_BY_TILE_DIR}\\MASK_{CSV_TILES_NAMES[key]}.jpg")


def creation_module():
    scale_px = 1 / SCALE_KM
    dC = DatasetCreator(MIN_CROP_AREA_SIZE_KM * scale_px, MAX_CROP_AREA_SIZE_KM * scale_px, SAMPLE_RESOLUTION, SCALE_KM)
    key = CSV_TILES_KEYS[2]
    dC.show_sample(load_zip_module(), TILES_NAMES[2], f"{TEMP_CRATERS_BY_TILE_DIR}\\MASK_{CSV_TILES_NAMES[key]}.jpg")


if __name__ == '__main__':
    open_zip_module()
    source_catalogue_module()
    # mask_module()
    # creation_module()
