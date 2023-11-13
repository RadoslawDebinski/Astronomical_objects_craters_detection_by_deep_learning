"""
    FILE PATHS CONSTANTS
"""

# Constant paths dict which is used to create folders tree of the whole project
CONST_PATH = {
    "data":     "data",
    "model":    "data\\model",
    "dataset":  "data\\dataset",

    "train":    "data\\dataset\\training",
    "trainIN":  "data\\dataset\\training\\input",
    "trainOUT": "data\\dataset\\training\\output",

    "valid":    "data\\dataset\\validation",
    "validIN":  "data\\dataset\\validation\\input",
    "validOUT": "data\\dataset\\validation\\output",

    "test":     "data\\dataset\\testing",
    "testIN":   "data\\dataset\\testing\\input",
    "testOUT":  "data\\dataset\\testing\\output",

    "cata":     "data\\dataset\\catalogs",
    "cataORG":  "data\\dataset\\catalogs\\original",
    "cataDIV":  "data\\dataset\\catalogs\\divided",

    "wac":      "data\\dataset\\wac",
    "wacORG":   "data\\dataset\\wac\\original",
    "wacMASK":  "data\\dataset\\wac\\masked",

    "marsORG":   "data\\transfer_learning\\original",
    "marsMASK":  "data\\transfer_learning\\masked",
    "marsIN":    "data\\transfer_learning\\input",
    "marsOUT":   "data\\transfer_learning\\output",

    "temp":     "data\\temp",
    "tempSRC":  "data\\temp\\source",
    "tempMars":  "data\\temp\\mars"

}
# Constant paths list which is used to clear dataset folders before creating the new one
CONST_PATH_CLEAR = [
    CONST_PATH["trainIN"], CONST_PATH["trainOUT"],
    CONST_PATH["validIN"], CONST_PATH["validOUT"],
    CONST_PATH["testIN"], CONST_PATH["testOUT"]
]

"""
    PROPERTIES OD ROBBINS MOON CRATERS CATALOGUE AND WAC TILES
"""
# Input 7zip fie name which contains all required data (WACs, catalog). Should be placed in CONST_PATH["tempSRC"] dir
INPUT_ZIP_NAME = "InputData.7z"

# Robbins catalogue of craters file name
MOON_CATALOGUE_NAME = "data_lunar_crater_database_robbins_2018.csv"
FIRST_COL_ID = "CRATER_ID"
# Constants for proper CSV and tales processing
CSV_TILES_NAMES = {
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

TILES_NAMES = [
    "WAC_GLOBAL_E300N2250_100M.tif",
    "WAC_GLOBAL_E300S2250_100M.tif",
    "WAC_GLOBAL_E300N3150_100M.tif",
    "WAC_GLOBAL_E300S3150_100M.tif",
    "WAC_GLOBAL_E300N0450_100M.tif",
    "WAC_GLOBAL_E300S0450_100M.tif",
    "WAC_GLOBAL_E300N1350_100M.tif",
    "WAC_GLOBAL_E300S1350_100M.tif"
]
COLS_NAMES_TO_ANALYZE = ["LAT_CIRC_IMG", "LON_CIRC_IMG", "LAT_ELLI_IMG", "LON_ELLI_IMG"]

# Bound for tiles with Equirectangular projection
TILES_BOUNDS = [(0,   60, 180, 270),
                (-60,  0, 180, 270),
                (0,   60, 270, 360),
                (-60,  0, 270, 360),
                (0,   60,   0,  90),
                (-60,  0,   0,  90),
                (0,   60,  90, 180),
                (-60,  0,  90, 180)]

# WAC tiles images constants
MOON_SCALE_KM = 0.1  # kilometers per pixel
MOON_SCALE_PX = 1 / MOON_SCALE_KM  # pixels per kilometer
RESOLUTION = 303.23  # pixels per degree

"""
    PROPERTIES OF ROBBINS MARS CATALOGUE AND MARS TILES
"""

MURRAY_LAB_URL = 'https://murray-lab.caltech.edu/CTX/V01/tiles/?fbclid=IwAR1QOxLwPgvvZ2ScBaI11hjX6mODBmdp-IesRkGBgO5HJKotpEqX6jBiFow'
MARS_CATALOGUE_NAME = 'Catalog_Mars_Release_2020_1kmPlus_FullMorphData.csv'
MARS_CATALOGUE_LONG = 'LON_CIRC_IMG'
MARS_CATALOGUE_LAT = 'LAT_CIRC_IMG'
MARS_CATALOGUE_DIAM = 'DIAM_CIRC_IMG'
MARS_TILE_DEG_SPAN = 4
MARS_SCALE_KM = 0.005  # kilometers per pixel

"""
    PROPERTIES OF CELESTIAL BODIES
"""

# Constants for Moon
MEAN_MOON_RADIUS_KM = 1737.05
LONGITUDE_MOON_CIRCUMFERENCE_KM = 10907
# Constant for Mars
MEAN_MARS_RADIUS_KM = 3389.5
LONGITUDE_MARS_CIRCUMFERENCE_KM = 21213.3

"""
    PROPERTIES OF SAMPLES
"""
# Properties of output images
MIN_CROP_AREA_SIZE_KM = 50
MAX_CROP_AREA_SIZE_KM = 50
SAMPLE_RESOLUTION = (256, 256)

# Other processing constants
CRATER_RIM_INTENSITY = 255
MOON_KERNEL_SIZE = 3
MARS_KERNEL_SIZE = 285

"""
    CONSTANTS FOR NEURAL NETWORK AND LEARNING PROCESS
"""

# Neural network architecture params
NET_PARAMS = {
    "in_channels":  1,
    "out_channels": 1,
    "filters_num":  64
}

# Optimizer params
OPTIM_PARAMS = {
    "learning_rate": 0.0005,
    "weight_decay":  10e-5
}

# Scheduler params
SCHED_PARAMS = {
    "t_max":   10,
    "eta_min": 0.001
}

# Training params
TRAIN_PARAMS = {
    "num_epochs": 15,
    "batch_size": 16,
    "save_interval_iter": 50
}
