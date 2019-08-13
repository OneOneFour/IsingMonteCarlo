import ast
import sys

from ising_feed_forward import plot_9_sample, plot_with_prediction, plot_train_val_acc, plot_train_val_loss
from tensorflow.python.keras import models, layers
from tensorflow.keras.callbacks import TensorBoard  # HAVE TO USE THIS NOT THE PYTHON ONE
import tensorflow as tf
from ising import IsingData
import neptune_tensorboard as neptune_tb
import neptune
from datetime import datetime
import numpy as np
file = None
files = [
    "../c++/batch_0_atcrit06082019.json",
    "../c++/batch_1_atcrit06082019.json",
    "../c++/batch_2_atcrit06082019.json"
]
PARAMS = {
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
    "epochs": 50,
    "batch_size": 64,
    "periodic_padding": False
}


def periodic_pad_numpy(x):
    assert x.shape[1] == x.shape[2]
    size = x.shape[1]
    b = np.eye(1, size, size - 1, dtype=np.float32)
    b = np.append(b, np.identity(size), axis=0)
    b = np.append(b, np.eye(1, size), axis=0)
    p_t = np.transpose(b)
    res = np.einsum("ijkl,km->ijml", np.einsum("mj,ijkl->imkl", b, x), p_t)
    return res


def periodic_pad(x):
    # Can use  p X pT
    assert x.shape[1] == x.shape[2]
    size = x.shape[1]
    b = np.eye(1, size, size - 1, dtype=np.float32)
    b = np.append(b, np.identity(size), axis=0)
    b = np.append(b, np.eye(1, size), axis=0)
    p = tf.constant(b, dtype=tf.float32)
    p_t = tf.transpose(p)
    res = tf.einsum("ijkl,km->ijml", tf.einsum("mj,ijkl->imkl", p, x), p_t)
    return res


def get_convolutional_network(use_periodic_pad=False):
    model = models.Sequential()
    if use_periodic_pad:
        model.add(layers.Lambda(periodic_pad))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model)
    return model


if __name__ == '__main__':
    if sys.argv[1] == "neptune":
        neptune.init(project_qualified_name="OneOneFour/Ising-Model")
        neptune_tb.integrate_with_tensorflow()
        ttf = IsingData(train_ratio=1, test_ratio=0.5, validation_ratio=0.20)
        if file:
            ttf.load(file)
        elif files:
            ttf.load_arr(files)
        else:
            raise ValueError("Nothing provided to load")
        (train_image, train_label), (test_image, test_label), (val_image, val_label) = ttf.get_data()

        # normalise and reshape

        train_image = train_image.reshape((len(train_image), 50, 50, 1))
        test_image = test_image.reshape((len(test_image), 50, 50, 1))
        val_image = val_image.reshape((len(val_image), 50, 50, 1))
        exp_name = f"Convolutional {file} {datetime.now().strftime('%Y_%m_%d')}"
        with neptune.create_experiment(name=exp_name, params=PARAMS) as exp:
            # if file:
            #     exp.send_artifact(file)
            # elif files:
            #     for f in files:
            #         exp.send_artifact(f)

            logdir = "..\\logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
            callback = TensorBoard(log_dir=logdir)  # Make sure to save callback as a regular variable
            model = get_convolutional_network(exp.get_parameters()['periodic_padding'])
            model.compile(optimizer=exp.get_parameters()['optimizer'],
                          loss=exp.get_parameters()['loss'],
                          metrics=ast.literal_eval(exp.get_parameters()['metrics']))
            model.fit(train_image, train_label, epochs=PARAMS['epochs'], validation_data=(val_image, val_label),
                      callbacks=[callback], batch_size=PARAMS['batch_size'])
            loss, acc = model.evaluate(test_image, test_label)
            print(f"Model accuracy: {acc}")
            exp.send_text("test-accuracy", acc)
            exp.send_text("test-loss", loss)
            weights_name = f"../weights/convolutional_weights {datetime.now().strftime('%Y_%m_%d %H_%M')}.h5"
            model.save_weights(weights_name)
            exp.send_artifact(weights_name)
    else:
        ttf = IsingData(train_ratio=1, test_ratio=0.5, validation_ratio=0.5)
        if file:
            ttf.load(file)
        elif files:
            ttf.load_arr(files)
        else:
            raise ValueError("Nothing provided to load")
        (train_image, train_label), (test_image, test_label), (val_image, val_label) = ttf.get_data()

        # normalise and reshape
        logdir = "logs/convo/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        train_image = train_image.reshape((len(train_image), 50, 50, 1))
        test_image = test_image.reshape((len(test_image), 50, 50, 1))
        val_image = val_image.reshape((len(val_image), 50, 50, 1))

        model = get_convolutional_network(PARAMS['periodic_padding'])
        model.compile(optimizer=PARAMS['optimizer'],
                      loss=PARAMS['loss'],
                      metrics=PARAMS['metrics'])
        model.fit(train_image, train_label, epochs=PARAMS['epochs'], validation_data=(val_image, val_label),
                  batch_size=PARAMS['batch_size'])
        loss, acc = model.evaluate(test_image, test_label)
        print(f"Model accuracy: {acc}")
