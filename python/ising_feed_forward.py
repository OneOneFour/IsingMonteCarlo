import numpy as np
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
from ising import TestTrainSetGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

file = "../json_dumps/batch_0_testing.json"

def plot_9_sample(images, labels):
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(fig.axes):
        ax.imshow(images[i], cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Label: {'Supercritical' if labels[i] > 0 else 'Subcritical'}")

    plt.show()


def plot_with_prediction(image, true_label, output):
    plt.imshow(image, cmap="binary")
    plt.title(f"Label Supercritical?:{bool(true_label)}")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"Prediction:{'Supercritical' if output > 0.5 else 'Subcritical'} ({output * 100})")
    plt.show()


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
    model.add(layers.Flatten(input_shape=(50, 50,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    history = model.fit(training_data, training_labels, epochs=epochs, batch_size=64,
                        validation_data=(validation_data, validation_labels))
    hist_dict = history.history
    plot_train_val_loss(hist_dict, epochs)
    plot_train_val_acc(hist_dict, epochs)
    return model, hist_dict


def feed_forward(training_data, training_labels, validation_data, validation_labels, epochs):
    tensorboard = TensorBoard(log_dir=f"logs\{time()}", )
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(50, 50,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    history = model.fit(training_data, training_labels, epochs=epochs, batch_size=64,
                        validation_data=(validation_data, validation_labels),
                        callbacks=[tensorboard])
    hist_dict = history.history
    plot_train_val_loss(hist_dict, epochs)
    plot_train_val_acc(hist_dict, epochs)

    return model, hist_dict


if __name__ == '__main__':
    ttsg = TestTrainSetGenerator()
    ttsg.load(file)
    (train_images, train_labels), (test_images, test_labels) = ttsg.get_data()

    train_images = (train_images + 1) / 2
    test_images = (test_images + 1) / 2


    split_at = len(train_images) // 5
    plot_9_sample(test_images, test_labels)

    validation_image = train_images[:split_at]
    validation_labels = train_labels[:split_at]

    partial_train_img = train_images[split_at:]
    partial_train_labels = train_labels[split_at:]

    model, hist_dict = feed_forward(partial_train_img, partial_train_labels, validation_image, validation_labels, 20)

    loss, acc = model.evaluate(test_images, test_labels)
    print(f"Model Accuracy on test set:{acc}")
