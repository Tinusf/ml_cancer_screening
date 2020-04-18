import matplotlib.pyplot as plt


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
