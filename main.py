from tensorflow.keras import layers, models
from data_loader import read_file, split_data
from PIL import Image
import numpy as np


def draw_image(numpy_3d_array):
    im = Image.fromarray(numpy_3d_array.astype(np.uint8))
    im.show()


def shuffle_data(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def main():
    X_data, y_data = read_file("data/skin/hmnist_28_28_RGB.csv")
    X_data, y_data = shuffle_data(X_data, y_data)
    y_data = np.array([1 if y in [1,5,6] else 0 for y in y_data])
    # draw_image(X_data[0])
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_data, y_data)
    model = models.Sequential()
    # TODO: tweak these hyperparams.
    model.add(layers.Conv2D(filters=28, kernel_size=(3, 3), activation='tanh', input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(112, (3, 3), activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(512, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Should we use validation datasets here? Or maybe cross-validation?
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.05)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print("Testing accuracy", test_acc)
    print("Testing loss", test_loss)

if __name__ == '__main__':
    main()
