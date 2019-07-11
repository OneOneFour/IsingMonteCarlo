import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from ising import TestTrainSetGenerator

if __name__ == '__main__':
    ttfing = TestTrainSetGenerator()
    ttfing.load(f"dump_test.json")

    (train_images, train_labels), (test_images, test_labels) = ttfing.get_data()

    # TODO: Maybe in future normalise the image input such that (1,-1) -> (1,0) probably needed to use ReLU further down the pipeline
    train_images = train_images/2 + 1
    test_images = test_images/2 + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(50, 50)),
        tf.keras.layers.Dense(100, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer="adam",  # Use ADAM as the optimizer as choice for now until there is any reason not too
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    model.fit(train_images, train_labels, epochs=5)

    test_lost, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy:{test_acc}")
