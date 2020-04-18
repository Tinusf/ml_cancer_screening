import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects, plot_model
from tensorflow.keras import callbacks, optimizers
from data_utils import save_history, load_history
from collections import defaultdict
import tensorflow_addons.metrics as metrics
import datetime
import config


class SkinCancerClassifier:
    def __init__(self, X_train=None, y_train=None, X_val=None, y_val=None):
        if config.WHICH_MODEL_LOAD is not None:
            self.load_saved_model()
        else:
            self.model = self.create_model()
            self.train_model(X_train, y_train, X_val, y_val)
            if config.DEBUG:
                plot_model(self.model, "model.png", show_shapes=True)

    def load_saved_model(self):
        get_custom_objects().update(
            {"swish": layers.Activation(self.swish), "F1Score":
                self.get_f1_score_metric()})
        custom_objects = {"swish": self.swish}
        model = load_model(f"{config.FOLDER_SAVE_MODEL_PATH}/{config.WHICH_MODEL_LOAD}",
                           custom_objects)
        history = load_history("latest")
        self.model = model
        self.history = history

    def predict(self, X_values):
        return self.model.predict(X_values).squeeze()

    @staticmethod
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

    @staticmethod
    def get_f1_score_metric():
        return metrics.F1Score(num_classes=config.NUMBER_OF_CLASSES, average="micro",
                               threshold=0.5)

    @staticmethod
    def swish(x):
        return K.sigmoid(x) * x


    def create_model(self):
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(28, 28, 3)))
        model.add(
            layers.Conv2D(
                filters=28, kernel_size=(5, 5), padding="same", activation="relu",
                input_shape=(28, 28, 3), strides=(2, 2)
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config.DROPOUT_PROB))

        model.add(layers.Conv2D(56, (5, 5), padding="same", activation="relu", strides=(2, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config.DROPOUT_PROB))

        model.add(layers.Conv2D(112, (5, 5), padding="same", activation="relu", strides=(2, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config.DROPOUT_PROB))

        model.add(layers.Flatten())

        model.add(layers.Dense(256))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.swish))
        model.add(layers.Dropout(config.DROPOUT_PROB))

        model.add(layers.Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.swish))
        model.add(layers.Dropout(config.DROPOUT_PROB))

        model.add(layers.Dense(256))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.swish))
        model.add(layers.Dropout(config.DROPOUT_PROB))

        model.add(layers.Dense(config.NUMBER_OF_CLASSES))
        model.add(layers.Activation("softmax"))
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.003),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", self.get_f1_score_metric()]
        )

        return model

    def train_model(self, X_train, y_train, X_val, y_val, save=True):
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
        )

        class_weights = self.get_class_weights(y_train)

        callbacks_list = []

        # Save the best validation accuracy model.
        checkpoint_val_acc = callbacks.ModelCheckpoint(f"{config.FOLDER_SAVE_MODEL_PATH}/best_val_acc.h5",
                                                       monitor='val_accuracy', save_best_only=True,
                                                       mode='max')
        callbacks_list.append(checkpoint_val_acc)

        # Save the best F1 score model.
        checkpoint_f1 = callbacks.ModelCheckpoint(f"{config.FOLDER_SAVE_MODEL_PATH}/best_f1.h5",
                                                  monitor='val_f1_score',
                                                  save_best_only=True, mode='max')

        callbacks_list.append(checkpoint_f1)

        # Save the best training accuracy (probably overfitted)
        checkpoint_acc = callbacks.ModelCheckpoint(f"{config.FOLDER_SAVE_MODEL_PATH}/best_acc.h5", monitor='accuracy',
                                                   save_best_only=True, mode='max')
        callbacks_list.append(checkpoint_acc)

        # Save the least Loss
        checkpoint_loss = callbacks.ModelCheckpoint(f"{config.FOLDER_SAVE_MODEL_PATH}/best_loss.h5", monitor='loss',
                                                    save_best_only=True, mode='min')
        callbacks_list.append(checkpoint_loss)

        # Save the least validation loss
        checkpoint_val_loss = callbacks.ModelCheckpoint(f"{config.FOLDER_SAVE_MODEL_PATH}/best_val_loss.h5",
                                                        monitor='val_loss', save_best_only=True,
                                                        mode='min')
        callbacks_list.append(checkpoint_val_loss)

        log_dir = f"{config.FOLDER_SAVE_MODEL_PATH}/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks_list.append(tensorboard_callback)

        if config.USE_EARLY_STOPPING:
            callbacks_list.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=10))

        datagen.fit(X_train)

        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
            steps_per_epoch=len(X_train) / config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            validation_steps=len(X_val) / config.BATCH_SIZE,
            callbacks=callbacks_list,
            class_weight=class_weights if config.USE_CLASS_WEIGHTS else None
        )
        if save:
            self.model.save(f"{config.FOLDER_SAVE_MODEL_PATH}/saved_model.h5")
            save_history(self.history)
