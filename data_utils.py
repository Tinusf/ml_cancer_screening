import csv
import pandas as pd
import os
from tensorflow.keras import callbacks, optimizers
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import config


def recreate_image(list_of_intensities):
    """
    :param list_of_intensities: Should be a list containing 28*28*3 (2352) numbers. The first
    three numbers are the R, B and B values of the first pixel. etc etc.
    :return: A 3d numpy array of height 28 and width 28 and each pixel being the 3 RGB values.
    """
    list_of_intensities = np.array(list_of_intensities)

    return list_of_intensities.reshape((28, 28, 3))


def read_file(file_name):
    # Reads the file and returns two numpy arrays, the first containing the feature-dataset and
    # the second list containing the labels.
    X_data = []
    y_data = []
    with open(file_name, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if row[0] == "pixel0000":
                # This is the very first row, which just described the attributes. So skip this.
                continue
            label = int(row[-1])
            list_1d_X = row[:-1]
            list_1d_X = [int(x) for x in list_1d_X]
            numpy_3d_array = recreate_image(list_1d_X)
            X_data.append(numpy_3d_array)
            y_data.append(label)
    return np.array(X_data), np.array(y_data)


def oversample(x, y):
    """
    This method oversamples the x and y so the under-represented classes gets duplicates.
    """
    image_shape = x.shape[1:]
    flatten_size = np.product(image_shape)
    x = x.reshape(x.shape[0], flatten_size)
    rus = RandomOverSampler(random_state=42)
    x, y = rus.fit_resample(x, y)
    x = x.reshape(x.shape[0], *image_shape)
    return x, y


def save_history(history):
    history_df = pd.DataFrame(history.history)

    if not os.path.exists(f"{config.FOLDER_SAVE_MODEL_PATH}/history"):
        os.mkdir(f"{config.FOLDER_SAVE_MODEL_PATH}/history")

    # Overwrite latest or rename all previous histories?

    with open(f"{config.FOLDER_SAVE_MODEL_PATH}/history/history_latest.csv", "w") as f:
        history_df.to_csv(f)


def load_history(version="latest"):
    file_name = f"{config.FOLDER_SAVE_MODEL_PATH}/history/history_" + version + ".csv"
    history = dict()
    if os.path.exists(file_name):
        with open(file_name, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            labels = []
            labels_read = False
            i = 0
            for row in reader:
                for e in row:
                    if e == "":
                        continue
                    if not labels_read:
                        labels.append(e)
                        history[e] = []
                        continue
                    if i == 0:
                        i = i + 1
                        continue
                    history[labels[i - 1]].append(float(e))
                    i = i + 1
                labels_read = True
                i = 0
    h = callbacks.History()
    h.history = history
    return h
