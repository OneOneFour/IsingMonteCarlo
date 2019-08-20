import sys

import numpy as np
from tensorflow.keras import layers, models, regularizers, callbacks
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ising import IsingData
import neptune_tensorboard as neptune_tb
import neptune
from datetime import datetime
import os

PARAMS = {
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "epochs": 50,
    "metrics": ["accuracy"],
    "batch_size": 64,
    "l2_regularization_weight": 0.01,
    "layer_dropout": 0.3,
    "layer_width": 128,
    "layer_depth": 3,
    "randomize_spins": True
}


class NeptuneCallback(callbacks.Callback):

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        self.experiment.send_metric('epoch_acc', logs['acc'])
        self.experiment.send_metric('epoch_loss', logs['loss'])
        self.experiment.send_metric('epoch_val_acc', logs['val_acc'])
        self.experiment.send_metric('epoch_val_loss', logs['val_loss'])


def plot_9_sample(images, labels):
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(fig.axes):
        ax.imshow(images[i], cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Label: {'Supercritical' if labels[i] > 0 else 'Subcritical'}")

    plt.show()


def plot_9_with_prediction(image, true_label, output):
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(fig.axes):
        ax.imshow(image[i], cmap="binary")
        ax.set_title(f"Label Supercritical?:{bool(true_label[i])}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Prediction:{'Supercritical' if output[i] > 0.5 else 'Subcritical'} ({output[i] * 100})")
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


def feed_forward(training_data, training_labels, validation_data, validation_labels, callback, size):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(size, size,)))

    for layer in range(PARAMS['layer_depth']):
        model.add(layers.Dense(PARAMS["layer_width"], activation='relu',
                               kernel_regularizer=regularizers.l2(PARAMS["l2_regularization_weight"])))

    model.add(layers.Dropout(PARAMS['layer_dropout']))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=PARAMS['optimizer'], loss=PARAMS['loss'], metrics=PARAMS['metrics'])
    history = model.fit(training_data, training_labels, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'],
                        validation_data=(validation_data, validation_labels),
                        callbacks=[callback])
    hist_dict = history.history
    # plot_train_val_loss(hist_dict, PARAMS['epochs'])
    # plot_train_val_acc(hist_dict, PARAMS['epochs'])
    print(model.summary())
    return model, hist_dict


def execute_feed_forward(head, tail, plotspectrum=True, runneptune=True, use_max=False):
    neptune.init("OneOneFour/Ising-Model")
    neptune_tb.integrate_with_tensorflow()
    ttsg = IsingData(train_ratio=5)
    ttsg.load_json(tail)
    if runneptune:
        exp = neptune.create_experiment(name=f"DFFN on {ttsg.size}x{ttsg.size} on file {tail}", params=PARAMS)
    if plotspectrum:
        e_overlap = ttsg.plot_energy_spectrum(20, "energy_spectrum.png")
        m_overlap = ttsg.plot_magnetization_spectrum(20, "magnetization_spectrum.png")
        if runneptune:
            energy_spectrum_img = Image.open("energy_spectrum.png")
            magnetization_spectrum_img = Image.open("magnetization_spectrum.png")

            exp.send_image("energy-spectrum", energy_spectrum_img)
            exp.send_image("magnetization-spectrum", magnetization_spectrum_img)

            exp.send_metric("energy-overlap", e_overlap)
            exp.send_metric("mag-overlap", m_overlap)

    (train_images, train_labels), (test_images, test_labels), (val_image, val_data) = ttsg.get_data()

    if PARAMS["randomize_spins"]:
        train_images = np.array([t * -1 if np.random.uniform(0, 1) > 0.5 else t for t in train_images])
        test_images = np.array([t * -1 if np.random.uniform(0, 1) > 0.5 else t for t in test_images])
        val_image = np.array([t * -1 if np.random.uniform(0, 1) > 0.5 else t for t in val_image])

    train_images = (train_images + 1) / 2
    test_images = (test_images + 1) / 2
    val_image = (val_image + 1) / 2

    callback = callbacks.TensorBoard(log_dir=f"logs\\ffn\\{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model, hist_dict = feed_forward(train_images, train_labels, val_image, val_data, callback, ttsg.size)

    if plotspectrum:
        pred_label = model.predict(test_images[:9])
        plot_9_with_prediction(test_images[:9], test_labels[:9], pred_label)

    max_acc = max(hist_dict["val_acc"])

    loss, acc = model.evaluate(test_images, test_labels)

    print(f"Model Accuracy on test set:{acc}")
    if runneptune:
        exp.send_artifact(tail)
        exp.send_text("test-accuracy", str(acc))
        exp.send_metric("max_acc", max_acc)
        exp.send_text("test-loss", str(loss))
        exp.send_text("file-name", tail)
        name = f"FFN_weights {datetime.now().strftime('%Y_%m_%d %H_%M')}.h5"
        model.save_weights(name)
        exp.send_artifact(name)
        exp.stop()
    if use_max:
        return loss, max_acc
    else:
        return loss, acc


def test_width(head, tail):
    min_width = 2
    max_width = 128
    widths = np.linspace(min_width, max_width, 32)
    acc = [0] * len(widths)
    loss = [0] * len(widths)
    for i, w in enumerate(widths):
        PARAMS["layer_width"] = w
        loss[i], acc[i] = execute_feed_forward(head, tail, plotspectrum=False, runneptune=False)
    import matplotlib

    plt.plot(widths, acc, label="Accuracy (Testing)")
    plt.ylabel("Accuracy")
    plt.xlabel("Network peak width")
    plt.savefig("Accuracy_vs_networkwidth")
    plt.show()
    return acc


def test_depth(head, tail):
    min_depth = 1
    max_depth = 20
    depths = np.linspace(min_depth, max_depth, 20, dtype=np.int32)
    acc = [0] * len(depths)
    loss = [0] * len(depths)
    for i, d in enumerate(depths):
        PARAMS["layer_depth"] = d
        loss[i], acc[i] = execute_feed_forward(head, tail, plotspectrum=False, runneptune=False)
    plt.plot(depths, acc, label="Accuracy (Testing)")
    plt.ylabel("Accuracy")
    plt.xlabel("Network depth")
    plt.savefig("Accuracy_vs_networkdepth.png")
    plt.show()
    return acc


def test_both(head, tail):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    min_depth = 1
    max_depth = 10
    min_width = 2
    max_width = 128

    width_steps = 10
    depth_steps = 10
    depths = np.linspace(min_depth, max_depth, depth_steps, dtype=np.int32)
    widths = np.linspace(min_width, max_width, width_steps, dtype=np.int32)

    dv, wv = np.meshgrid(depths, widths)
    acc = np.zeros((width_steps, depth_steps))
    for i in range(depth_steps):
        for j in range(width_steps):
            PARAMS["layer_depth"] = dv[j, i]
            PARAMS["layer_width"] = wv[j, i]
            loss_ij, acc_ij = execute_feed_forward(head, tail, plotspectrum=False, runneptune=False)
            acc[j, i] = acc_ij
    ax.plot_surface(dv, wv, acc, cmap="jet")
    ax.set_xlabel("Network depth")
    ax.set_ylabel("Network width")
    ax.set_zlabel("Accuracy")
    ax.set_title(f"{tail}")
    plt.savefig(f"Accuracy_vs_network.png")
    plt.show()
    return acc


if __name__ == '__main__':
    file = input("Enter JSON file to load into FFN")
    head, tail = os.path.split(file)
    os.chdir(os.path.join(os.getcwd(), head))
    print(os.getcwd())
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_width":
            test_width(head, tail)
        elif sys.argv[1] == "test_depth":
            test_depth(head, tail)

        elif sys.argv[1] == "test_both":
            test_both(head, tail)
    else:

        execute_feed_forward(head, tail)
