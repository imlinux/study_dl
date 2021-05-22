import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt;
import numpy as np


def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(1280, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    predictions = model.predict(test_images)
    print("p=", predictions[0])
    print(np.argmax(predictions[0]), test_labels[0])


if __name__ == '__main__':
    main()
