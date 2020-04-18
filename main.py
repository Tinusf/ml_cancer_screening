import draw
import data_utils
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import config
from SkinCancerClassifier import SkinCancerClassifier


def main():
    X_data, y_data = data_utils.read_file("data/skin/hmnist_28_28_RGB.csv")
    X_data = X_data.astype('float64')

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                        test_size=config.TEST_SIZE,
                                                        random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=config.VALIDATION_SIZE,
                                                      random_state=42)
    if config.USE_OVERSAMPLING:
        # Oversample the training set.
        X_train, y_train = data_utils.oversample(X_train, y_train)

        # Oversample the validation set.
        X_val, y_val = data_utils.oversample(X_val, y_val)

        # Oversample the test set.
        X_test, y_test = data_utils.oversample(X_test, y_test)

    skin_cancer_classifier = SkinCancerClassifier(X_train, y_train, X_val, y_val)

    stats = skin_cancer_classifier.model.evaluate(X_test, y_test, verbose=2)
    print("Testing loss", stats[0])
    print("Testing accuracy", stats[1])
    if len(stats) == 3:
        print("F1 score", stats[2])

    y_pred = skin_cancer_classifier.model.predict(X_test)
    # Decode the one-hot vector.
    y_pred = np.argmax(y_pred, axis=1)
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    target_names = ['Actinic Keratoses', 'Basal cell carcinoma', 'Benign keratosis',
                    'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular skin lesions']
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names))

    if config.DEBUG:
        draw.plot_performance(skin_cancer_classifier.history)


if __name__ == "__main__":
    main()
