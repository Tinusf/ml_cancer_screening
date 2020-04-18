
# Either a string filepath or None if you want to retrain.
WHICH_MODEL_LOAD = "best_f1.h5"
DEBUG = True # TODO: rename
# How many epochs
EPOCHS = 1
BATCH_SIZE = 128
# Class weighting, in order to counter the effects of the imbalanced data.
USE_CLASS_WEIGHTS = False
USE_EARLY_STOPPING = False
NUMBER_OF_CLASSES = 7

VALIDATION_SIZE = 0.05
TEST_SIZE = 0.05

DROPOUT_PROB = 0.2

FOLDER_SAVE_MODEL_PATH = "saved_models"

USE_OVERSAMPLING = False