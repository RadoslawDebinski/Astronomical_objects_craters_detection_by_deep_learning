from csv_sorter import SourceTypeSeparator
from img_module import ImgAnalyzer
import re

source_dataset_dir = "C:\\ACIR-WETI\\Praca_Inzynierska\\Input_data\\pdsimage2.wr.usgs.gov_Individual_Investigations_moon_lro.kaguya_multi_craterdatabase_robbins_2018_data_lunar_crater_database_robbins_2018.csv"
split_craters_by_tile_dir = "C:\\ACIR-WETI\\Praca_Inzynierska\\Code_Section\\InputData"
tiles_names = {
    "00-1-\d{6}": "WAC_GLOBAL_P900S0000_LAT_-90_to_-60_LON____0_to_360",
    "01-1-\d{6}": "WAC_GLOBAL_P900N0000_LAT__60_to__90_LON____0_to_360",
    "02-1-\d{6}": "WAC_GLOBAL_E300N2250_LAT___0_to__60_LON__180_to_270",
    "03-1-\d{6}": "WAC_GLOBAL_E300S2250_LAT_-60_to___0_LON__180_to_270",
    "04-1-\d{6}": "WAC_GLOBAL_E300N3150_LAT___0_to__60_LON__270_to_360",
    "05-1-\d{6}": "WAC_GLOBAL_E300S3150_LAT_-60_to__0__LON__270_to_360",
    "06-1-\d{6}": "WAC_GLOBAL_E300N0450_LAT___0_to__60_LON_____0_to_90",
    "07-1-\d{6}": "WAC_GLOBAL_E300S0450_LAT_-60_to__0__LON_____0_to_90",
    "08-1-\d{6}": "WAC_GLOBAL_E300N1350_LAT___0_to__60_LON___90_to_180",
    "09-1-\d{6}": "WAC_GLOBAL_E300S1350_LAT_-60_to___0_LON___90_to_180",
}
first_col_id = "CRATER_ID"
cols_names_to_analyze = ["LAT_CIRC_IMG", "LON_CIRC_IMG", "LAT_ELLI_IMG", "LON_ELLI_IMG"]


def source_dataset_module():
    sTS = SourceTypeSeparator("00", source_dataset_dir, first_col_id, tiles_names, split_craters_by_tile_dir)
    sTS.split_craters_by_tile_id()
    sTS.analyze_split_crater_by_tile_id(cols_names_to_analyze)


img_path = "A:\\Inz_data\\WAC_GLOBAL_E300N2250_100M.tif"
longitude = 255.804
latitude = 49.4886


def images_module():
    iMGA = ImgAnalyzer()
    iMGA.img_load_convert(img_path)
    iMGA.place_craters_centers("WAC_GLOBAL_E300N2250_LAT___0_to__60_LON__180_to_270.csv")
    iMGA.img_analyze(iMGA.rgb_img)
    # iMGA.save_image("with_red_points_WAC_GLOBAL_E300N2250_LAT___0_to__60_LON__180_to_270.csv", iMGA.rgb_img)


if __name__ == '__main__':
    # Section of CSV dataset splitting and analysis
    # source_dataset_module()

    # Section of images loading, converting, processing and display
    images_module()




