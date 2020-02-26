import csv
import numpy as np


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
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
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
