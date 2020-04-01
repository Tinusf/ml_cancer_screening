import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects, plot_model
from tensorflow.keras import callbacks, optimizers
from data_loader import read_file, save_history, load_history
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow_addons.metrics as metrics
import datetime
from imblearn.over_sampling import RandomOverSampler

# Load the model saved to file instead of creating a new.
USE_SAVED_MODEL = False
DEBUG = False
# How many epochs
EPOCHS = 12000
BATCH_SIZE = 128
# Class weighting, in order to counter the effects of the imbalanced data.
USE_CLASS_WEIGHTS = False
USE_EARLY_STOPPING = False
NUMBER_OF_CLASSES = 7

VALIDATION_SIZE = 0.05

DROPOUT_PROB = 0.2


def draw_image(numpy_3d_array):
    im = Image.fromarray(numpy_3d_array.astype(np.uint8))
    im.show()


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
    model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2D(
            filters=28, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 3)
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(DROPOUT_PROB))

    model.add(layers.Conv2D(56, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(DROPOUT_PROB))

    model.add(layers.Conv2D(112, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(DROPOUT_PROB))

    model.add(layers.Flatten())

    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(swish))
    model.add(layers.Dropout(DROPOUT_PROB))

    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(swish))
    model.add(layers.Dropout(DROPOUT_PROB))

    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(swish))
    model.add(layers.Dropout(DROPOUT_PROB))

    model.add(layers.Dense(NUMBER_OF_CLASSES))
    model.add(layers.Activation("softmax"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", get_f1_score_metric()]
    )

    return model


def get_f1_score_metric():
    return metrics.F1Score(num_classes=NUMBER_OF_CLASSES, average="micro", threshold=0.5)


def train_model(model, X_train, y_train, save=True):
    """
    :param X_train: The features that should be trained on.
    :param y_train: The labels
    :param save: True if the newly trained model should be saved.
    :return: The newly trained model.
    """

    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.5,
        height_shift_range=0.5,
        brightness_range=[0.5, 1.0],
        shear_range=1.0,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VALIDATION_SIZE,
    )

    class_weights = get_class_weights(y_train)

    callbacks_list = []

    # Save the best validation accuracy model.
    checkpoint_val_acc = callbacks.ModelCheckpoint("saved_models/best_val_acc.h5",
                                                   monitor='val_accuracy', save_best_only=True,
                                                   mode='max')
    callbacks_list.append(checkpoint_val_acc)

    # Save the best F1 score model.
    checkpoint_f1 = callbacks.ModelCheckpoint("saved_models/best_f1.h5", monitor='val_f1_score',
                                              save_best_only=True, mode='max')

    callbacks_list.append(checkpoint_f1)

    # Save the best training accuracy (probably overfitted)
    checkpoint_acc = callbacks.ModelCheckpoint("saved_models/best_acc.h5", monitor='accuracy',
                                               save_best_only=True, mode='max')
    callbacks_list.append(checkpoint_acc)

    # Save the least Loss
    checkpoint_loss = callbacks.ModelCheckpoint("saved_models/best_loss.h5", monitor='loss',
                                                save_best_only=True, mode='min')
    callbacks_list.append(checkpoint_loss)

    # Save the least validation loss
    checkpoint_val_loss = callbacks.ModelCheckpoint("saved_models/best_val_loss.h5",
                                                    monitor='val_loss', save_best_only=True,
                                                    mode='min')
    callbacks_list.append(checkpoint_val_loss)

    log_dir = "saved_models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list.append(tensorboard_callback)

    if USE_EARLY_STOPPING:
        callbacks_list.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=10))

    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='training'),
        steps_per_epoch=len(X_train) * (1 - VALIDATION_SIZE) / BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='validation'),
        validation_steps=len(X_train) * VALIDATION_SIZE / BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None
    )
    if save:
        model.save("saved_models/saved_model.h5")
        save_history(history)
    return model, history


def get_saved_model():
    get_custom_objects().update(
        {"swish": layers.Activation(swish), "F1Score": get_f1_score_metric()})
    custom_objects = {"swish": swish}
    model = load_model("saved_models/best_f1.h5", custom_objects)
    history = load_history("latest")
    return model, history


def predict(model, X_values):
    return model.predict(X_values).squeeze()


def plot_performance(history):
    x_axis = list(range(len(history.history["accuracy"])))
    plt.plot(x_axis, history.history["accuracy"], "b", label="Training accuracy")
    plt.plot(x_axis, history.history["val_accuracy"], "r", label="Validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    x_axis = list(range(len(history.history["loss"])))
    plt.plot(x_axis, history.history["loss"], "b", label="Training loss")
    plt.plot(x_axis, history.history["val_loss"], "r", label="Validation loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def oversample(x, y):
    image_shape = x.shape[1:]
    flatten_size = np.product(image_shape)
    x = x.reshape(x.shape[0], flatten_size)
    rus = RandomOverSampler(random_state=42)
    x, y = rus.fit_resample(x, y)
    x = x.reshape(x.shape[0], *image_shape)
    return x, y


def main():
    X_data, y_data = read_file("data/skin/hmnist_28_28_RGB.csv")
    X_data = X_data.astype('float64')

    X_data, y_data = oversample(X_data, y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2,
                                                        random_state=42)

    if USE_SAVED_MODEL:
        model, history = get_saved_model()
    else:
        model = create_model()
        model, history = train_model(model, X_train, y_train)
        if DEBUG:
            plot_model(model, "model.png", show_shapes=True)

    stats = model.evaluate(X_test, y_test, verbose=2)
    print("Testing loss", stats[0])
    print("Testing accuracy", stats[1])
    if len(stats) == 3:
        print("F1 score", stats[2])

    y_pred = model.predict(X_test)
    # Decode the one-hot vector.
    y_pred = np.argmax(y_pred, axis=1)
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    target_names = ['Actinic Keratoses', 'Basal cell carcinoma', 'Benign keratosis',
                    'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular skin lesions']
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names))

    if DEBUG:
        plot_performance(history)


if __name__ == "__main__":
    main()
