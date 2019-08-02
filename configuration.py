# === import configurarion

# ==== MODEL CONFIGURATION
NUM_CLASSES = 2
TRAIN_BATCH_SIZE = 32
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
VALIDATE = 'store_false'
SAVE_WEIGHTS_PATH = "PATH_TO_SAVE_WEIGHTS"
EPOCHS = 5
LOAD_WEIGHTS = ""
OPTIMISER_NAME = "adadelta"
MODEL_NAME = "vgg_unet"
VGG_WEIGHTS_PATH = "PATH_TO_VGG_WEIGHT_PATH/vgg16_weights_th_dim_ordering_th_kernels.h5"
SS_MODEL_PATH = "/run/media/sadeghip/MyStorage/LG_EarCnt_SS_FT/MODELS/SS_MODEL/model_SS.hdf5"

# ==== TRAINING CONFIGURATION
TRAIN_IMAGES_PATH = "PATH_TO_TRAIN_IMAGES"
TRAIN_SEGMENTATION_PATH = "PATH_TO_SEGMENTED_IMAGES"

# ==== VALIDATION CONFIGURATION
VAL_IMAGES_PATH = "PATH_TO_VALIDATION_IMAGES"
VAL_SEGS_PATH = "PATH_TO_VALIDATION_SEGMENTED_IMAGES"
VAL_BATCH_SIZE = 32

# define the path to the dataset mean
# DATASET_MEAN = "/run/media/sadeghip/MyStorage/LG_EarCnt_SS_FT/MODELS/FT_MODEL/noRatio/color_mean.json"

# ==== SLIC SETTINGS
SLIC_SEGMENT = 2500
SLIC_COMPACT = 55
SLIC_SIGMA = 1

NOISE_Thr = 50
watershed_Thr = 45

# ====== TEST IMAGES
EPOCHS_TEST = 1

TEST_IMAGES_PATH = "PATH_TO_TEST_IMAGES"
SAVE_RESULTS_PATH = "PATH_TO_SAVE_RESULTS"

