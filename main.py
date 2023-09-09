from csv_sorter_module import SourceTypeSeparator
from mask_module import MaskCreator
from dataset_creation_module import DatasetCreator
import re

# TODO: Remade mask module to create mask files for all tiles
# TODO: Make gitignore as it should be ;)
# TODO: Remade this repo following PLUG & PLAY way
# TODO: Make run_pipeline in dataset_creation_module
# TODO: Test if dataset creator works

# Input and temp directories / names
INPUT_ZIP_PATH = "InputData.7z"
CRATERS_CATALOGUE_NAME = "pdsimage2.wr.usgs.gov_Individual_Investigations_moon_lro" \
                         ".kaguya_multi_craterdatabase_robbins_2018_data_lunar_crater_database_robbins_2018.csv"
TEMP_CRATERS_BY_TILE_DIR = "tempData"
# Constants for proper CSV processing
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
FIRST_COL_ID = "CRATER_ID"
COLS_NAMES_TO_ANALYZE = ["LAT_CIRC_IMG", "LON_CIRC_IMG", "LAT_ELLI_IMG", "LON_ELLI_IMG"]
# WAC tiles images constants
IMG_PATH = "C:\ACIR-WETI\Praca_Inzynierska\dataset_module_input\WAC_GLOBAL_E300N2250_100M.tif"
SCALE_KM = 0.1  # kilometers per pixel
RESOLUTION = 303.23  # pixels per degree
# End dataset properties
MIN_CROP_AREA_SIZE_KM = 50
MAX_CROP_AREA_SIZE_KM = 100
SAMPLE_RESOLUTION = (512, 512)
# Moon constants
MEAN_MOON_RADIUS_KM = 1737.05
LONGITUDE_MOON_CIRCUMFERENCE_KM = 10907
# Other constants
CRATER_RIM_INTENSITY = 255


# CSV dataset splitting and analysis module handling
def source_catalogue_module():
    sTS = SourceTypeSeparator(INPUT_ZIP_PATH, CRATERS_CATALOGUE_NAME, FIRST_COL_ID, CSV_TILES_NAMES, TEMP_CRATERS_BY_TILE_DIR)
    sTS.split_craters_by_tile_id()
    # sTS.analyze_split_crater_by_tile_id(COLS_NAMES_TO_ANALYZE)


# Images loading, conversion, processing and mask creation module handling
def mask_module():
    iMGA = MaskCreator(SCALE_KM, MEAN_MOON_RADIUS_KM, LONGITUDE_MOON_CIRCUMFERENCE_KM, CRATER_RIM_INTENSITY)
    iMGA.img_load(IMG_PATH)
    iMGA.img_analyze()
    key = CSV_TILES_KEYS[2]
    iMGA.place_craters(f"{TEMP_CRATERS_BY_TILE_DIR}\\{CSV_TILES_NAMES[key]}.csv")
    iMGA.save_mask(f"{TEMP_CRATERS_BY_TILE_DIR}\\MASK_{CSV_TILES_NAMES[key]}.jpg")


def creation_module():
    scale_px = 1 / SCALE_KM
    dC = DatasetCreator(MIN_CROP_AREA_SIZE_KM * scale_px, MAX_CROP_AREA_SIZE_KM * scale_px, SAMPLE_RESOLUTION, SCALE_KM)
    key = CSV_TILES_KEYS[2]
    dC.show_sample(IMG_PATH, f"{TEMP_CRATERS_BY_TILE_DIR}\\MASK_{CSV_TILES_NAMES[key]}.jpg")


if __name__ == '__main__':
    source_catalogue_module()
    # mask_module()
    # creation_module()




