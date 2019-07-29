import numpy as np
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
from ising import TestTrainSetGenerator


def plot_9_sample():
    pass


def plot_with_prediction():
    pass


def plot_train_val_loss(hist_dict, epochs):
    epoch_list = range(1, epochs + 1)
    loss_vals = hist_dict['loss']
    validation_loss_vals = hist_dict['val_loss']
    plt.plot(epoch_list, loss_vals, 'bo', label="Training loss")
    plt.plot(epoch_list, validation_loss_vals, 'b', label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_train_val_acc(hist_dict, epochs):
    epoch_list = range(1, epochs + 1)
    acc_vals = hist_dict['acc']
    validation_acc_vals = hist_dict['val_acc']
    plt.plot(epoch_list, acc_vals, 'bo', label="Training accuracy")
    plt.plot(epoch_list, validation_acc_vals, 'b', label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def perceptron_test(training_data, training_labels, validation_data, validation_labels, epochs):
    model = models.Sequential()
    model.add(layers.Dense(1, activation='sigmoid', input_shape=(training_data[0].shape,)))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    history = model.fit(training_data, training_labels, epochs=epochs, batch_size=64,
                        validation_data=(validation_data, validation_labels))
    hist_dict = history.history

    return model, hist_dict


def feed_forward(training_data, training_labels, validation_data, validation_labels, epochs):
    pass
