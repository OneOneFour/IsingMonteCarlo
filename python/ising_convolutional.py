from ising_feed_forward import plot_9_sample, plot_with_prediction, plot_train_val_acc, plot_train_val_loss
from tensorflow.python.keras import models, layers
import tensorflow as tf
from ising import TestTrainSetGenerator
from ising_feed_forward import NeptuneCallback
import neptune
import numpy as np

file = "../c++/batch_0_twobatch.json"


def periodic_pad(x):
    # Can use  p X pT
    assert x.shape[0] == x.shape[1]
    size = x.shape[1]
    b = np.eye(1, size, size - 1, dtype=np.float64)
    b = np.append(b, np.identity(size),axis=0)
    b = np.append(b, np.eye(1, size),axis=0)

    p = tf.constant(b)
    pt = tf.transpose(p)
    return tf.matmul(tf.matmul(p, x), pt)


def get_convolutional_network():
    model = models.Sequential([
        layers.Lambda(periodic_pad),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


if __name__ == '__main__':
    neptune.init("OneOneFour/Ising-Model")
    ttf = TestTrainSetGenerator(train_ratio=1, test_ratio=0.5, validation_ratio=0.5)
    ttf.load(file)
    (train_image, train_label), (test_image, test_label), (val_image, val_label) = ttf.get_data()

    # normalise and reshape

    train_image = train_image.reshape((len(train_image), 50, 50, 1))
    test_image = test_image.reshape((len(test_image), 50, 50, 1))
    val_image = val_image.reshape((len(val_image), 50, 50, 1))
    with neptune.create_experiment() as exp:
        model = get_convolutional_network()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        model.fit(train_image, train_label, epochs=40, validation_data=(val_image, val_label),
                  callbacks=[NeptuneCallback(exp)])

        loss, acc = model.evaluate(test_image, test_label)
        print(f"Model accuracy: {acc}")
