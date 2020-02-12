from tensorflow.keras import layers, models, preprocessing, backend as K
from data_loader import read_file, split_data
from PIL import Image
import numpy as np

USE_SAVED_MODEL = True


def draw_image(numpy_3d_array):
    im = Image.fromarray(numpy_3d_array.astype(np.uint8))
    im.show()


def shuffle_data(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def swish(x):
    return (K.sigmoid(x) * x)


def create_and_train_model(X_train, y_train, save=True):
    """
    :param X_train: The features that should be trained on.
    :param y_train: The labels
    :param save: True if the newly trained model should be saved.
    :return: The newly trained model.
    """
    model = models.Sequential()
    # TODO: tweak these hyperparams.
    model.add(
        layers.Conv2D(filters=28, kernel_size=(3, 3), activation=swish, input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation=swish))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(112, (3, 3), activation=swish))
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(128, activation=swish))
    model.add(layers.Dense(512, activation=swish))
    # model.add(layers.Dense(64, activation=swish))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.0],
        shear_range=1.0,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.05)

    datagen.fit(X_train)
    model.fit(datagen.flow(X_train, y_train, batch_size=128),
              steps_per_epoch=len(X_train) / 128, epochs=150)
    if save:
        model.save("saved_model.h5")
    return model


def main():
    X_data, y_data = read_file("data/skin/hmnist_28_28_RGB.csv")
    X_data, y_data = shuffle_data(X_data, y_data)
    # y_data = np.array([1 if y in [1,5,6] else 0 for y in y_data])
    # draw_image(X_data[0])
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_data, y_data)
    if USE_SAVED_MODEL:
        model = models.load_model("saved_model.h5")
    else:
        model = create_and_train_model(X_train, y_train)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print("Testing accuracy", test_acc)
    print("Testing loss", test_loss)


if __name__ == '__main__':
    main()
