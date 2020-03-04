from tensorflow.keras import layers, models, preprocessing, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects, plot_model
from tensorflow.keras import callbacks
from data_loader import read_file, save_history, load_history
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure

# from tensorflow.keras.applications.densenet import DenseNet121

# Load the model saved to file instead of creating a new.
USE_SAVED_MODEL = True
DEBUG = True
# How many epochs
EPOCHS = 10
BATCH_SIZE = 128
# Class weighting, in order to counter the effects of the inbalanced data.
USE_CLASS_WEIGHTS = False
USE_EARLY_STOPPING = False

target_names = [
    "Actinic Keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevi",
    "Melanoma",
    "Vascular skin lesions",
]


def draw_image(numpy_3d_array):
    im = Image.fromarray(numpy_3d_array.astype(np.uint8))
    im.show()


def create_hog(image):
    hog_image = hog(
        image,
        orientation=9,
        pixel_per_cell=(2, 2),
        cells_per_block=(1, 1),
        visualizer=False,
        multichannel=True,
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))
    return hog_image_rescaled


def swish(x):
    return K.sigmoid(x) * x


def get_class_weights(labels, num_of_classes=7):
    # The higher the weight the less common the class is.
    # Count how many of the different classes we have.
    counter = defaultdict(lambda: 0)
    for label in labels:
        counter[label] += 1
    total = sum(counter.values())

    class_weights = {}
    for i in range(0, num_of_classes):
        class_weights[i] = (1 / counter[i]) * (total) / num_of_classes

    return class_weights


def create_model():
    model = models.Sequential()
    # TODO: tweak these hyperparams.
    model.add(
        layers.Conv2D(
            filters=28, kernel_size=(3, 3), activation=swish, input_shape=(28, 28, 3)
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation=swish))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(112, (3, 3), activation=swish))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=swish))
    model.add(layers.Dense(512, activation=swish))
    model.add(layers.Dense(256, activation=swish))
    model.add(layers.Dense(7, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train, save=True):
    """
    :param X_train: The features that should be trained on.
    :param y_train: The labels
    :param save: True if the newly trained model should be saved.
    :return: The newly trained model.
    """

    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.0],
        shear_range=1.0,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.05,
    )

    class_weights = get_class_weights(y_train)

    if USE_EARLY_STOPPING:
        es = callbacks.EarlyStopping(monitor="accuracy")
    else:
        es = None

    datagen.fit(X_train)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) / BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=es if es is not None else None,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None,
    )
    if save:
        model.save("saved_model.h5")
        save_history(history)
    return model, history


def get_saved_model():
    get_custom_objects().update({"swish": layers.Activation(swish)})
    custom_objects = {"swish": swish}
    model = load_model("saved_model.h5", custom_objects)
    history = load_history("latest")
    return model, history


def predict(model, X_values):
    return model.predict(X_values).squeeze()


def plot_performance(history):
    plt.plot(history.history["accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def main():
    X_data, y_data = read_file("data/skin/hmnist_28_28_RGB.csv")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)

    # model = DenseNet121(include_top=False, weights='imagenet', input_tensor=None,
    #                     input_shape=(28, 28, 3), pooling=None, classes=1000)
    # model.compile(
    #     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )
    #  model, history = train_model(model, X_train, y_train)

    if USE_SAVED_MODEL:
        model, history = get_saved_model()
    else:
        model = create_model()
        model, history = train_model(model, X_train, y_train)
        if DEBUG:
            plot_model(model, "model.png", show_shapes=True)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Testing accuracy", test_acc)
    print("Testing loss", test_loss)

    y_pred = model.predict(X_test)
    # Decode the one-hot vector.
    y_pred = np.argmax(y_pred, axis=1)
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(
        classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names)
    )

    if DEBUG:
        plot_performance(history)


if __name__ == "__main__":
    main()
