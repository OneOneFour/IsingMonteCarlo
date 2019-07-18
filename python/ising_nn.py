import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from ising import TestTrainSetGenerator
from datetime import datetime as dt


def plot_image(image, label):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.colorbar()
    plt.title(f"Image is {'above critical temperature' if label else 'below the critical temperature'}")
    plt.show()


def plot_image_with_nn(image, label, prediction):
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Label Supercritical?:{bool(label)}")
    plt.subplot(1, 2, 2)
    plt.title(f"Prediction:{'Supercritical' if np.argmax(prediction) else 'Subcritical'} ({round(prediction[np.argmax(prediction)] * 100)}%)")
    plt.bar(range(2), prediction)
    plt.show()


if __name__ == '__main__':
    ttfing = TestTrainSetGenerator(0.9)
    ttfing.load(f"18-07-2019 15-29-58dump.json")

    (train_images, train_labels), (test_images, test_labels) = ttfing.get_data()

    # Normalise the spins which is important for activation functions
    train_images = (train_images + 1) / 2
    test_images = (test_images + 1) / 2

    plot_image(train_images[0], train_labels[0])

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(ttfing.size, ttfing.size)),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer="adam",  # Use ADAM as the optimizer as choice for now until there is any reason not too
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    # log_dir = f"logs\\fit\{dt.now().strftime('%Y%m%d-%H%M%S')}"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_images,
              train_labels,
              epochs=10,
              validation_data=(test_images, test_labels),
              # callbacks=[tensorboard_callback]
              )

    predict = model.predict(test_images)
    for i, img in enumerate(test_images[:10]):
        plot_image_with_nn(img, test_labels[i], predict[i])

    harder_test = TestTrainSetGenerator()
    harder_test.load(f"dump_testT00-1(old).json")
    (train_images_h, train_labels_h), (test_images_h, test_labels_h) = harder_test.get_data()
    train_images_h = (train_images_h + 1) / 2
    test_images_h = (test_images_h + 1) / 2
    hard_lost, hard_acc = model.evaluate(test_images_h, test_labels_h)
    print(f"Harder test accuracy:{hard_acc}")

    predict_h = model.predict(test_images_h)
    for i, img in enumerate(test_images_h[:10]):
        plot_image_with_nn(img, test_labels_h[i], predict_h[i])
