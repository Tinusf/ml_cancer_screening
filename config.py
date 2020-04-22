# Either a string filepath or None if you want to retrain.
WHICH_MODEL_LOAD = "best_val_loss.h5"
# Should the model be plotted.
PLOT_MODEL = True
# How many epochs to run.
EPOCHS = 1
# Batch size.
BATCH_SIZE = 128
# Class weighting, in order to counter the effects of the imbalanced data.
USE_CLASS_WEIGHTS = False
# If early stopping should be used.
USE_EARLY_STOPPING = False
# How many different classes there are.
NUMBER_OF_CLASSES = 7
# The ratio of data that should be used for validation.
VALIDATION_SIZE = 0.05
# Ratio of the data that should be used for testing.
TEST_SIZE = 0.05
# The probability of dropout for all layers.
DROPOUT_PROB = 0.2
# Which folder to save and load the model.
FOLDER_SAVE_MODEL_PATH = "saved_models"
# Should oversampling be used. This can be done in order to balance the data.
USE_OVERSAMPLING = False
# The learning rate for Adam.
LEARNING_RATE = 0.003
