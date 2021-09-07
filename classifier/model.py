import os
import os
import pathlib
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from classifier.layers import Rescaling
from folders import classifier_tree, classifier


class TreeClassifier:

    def __init__(self, saved_model=None):

        self.history = None
        self.rescale = Rescaling(1. / 255)
        self.data_dir = classifier_tree()
        self.data_dir = pathlib.Path(self.data_dir)
        self.classes = len(next(os.walk(self.data_dir))[1])
        self.batch_size = 200
        self.img_size = (250, 250)
        self.class_names = ["Birch", "Spruce", "Pine"]

        print("Initializing tree classifier...")
        print("Classes:", self.class_names)

        self.model = Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(250, 250, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.classes)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        if saved_model is not None:
            self.load_model(saved_model)

    def train(self, epochs):

        print("Starting training loop...")

        generator = ImageDataGenerator(
            rescale=1. / 255,
            brightness_range=[0.9, 1.1],
            shear_range=0.2,
            rotation_range=360,
            zoom_range=0.2,
            channel_shift_range=25,
            vertical_flip=True,
            horizontal_flip=True,
            validation_split=0.2)

        train_generator = generator.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset="training")

        validation_generator = generator.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset="validation")

        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            workers=5,
            validation_data=validation_generator,
            steps_per_epoch=train_generator.samples // self.batch_size - 1,
            validation_steps=round((train_generator.samples // self.batch_size) * 0.2))

    def plot_history(self):

        data = self.history.history
        epochs = len(data['accuracy'])

        plt.figure(figsize=(8, 8))

        plt.subplot(1, 2, 1)
        plt.plot(data['loss'], color='b', label="Training loss")
        plt.plot(data['val_loss'], color='r', label="Validation loss")
        plt.xticks(np.arange(0, epochs, 1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.legend(loc='lower right')
        plt.title('Training and Validation loss')

        plt.subplot(1, 2, 2)
        plt.plot(data['accuracy'], color='b', label="Training accuracy")
        plt.plot(data['val_accuracy'], color='r', label="Validation accuracy")
        plt.xticks(np.arange(0, epochs + 1, 1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.title('Training and Validation accuracy')
        plt.show()

    def preprocess_img(self, pil_img):

        resize_img = pil_img.resize(self.img_size, Image.NEAREST)
        img_array = keras.preprocessing.image.img_to_array(resize_img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = self.rescale(img_array)

        return img_array

    def classify_batch_of_trees(self, img_list, batch_size=400, score_threshold=0.70):
        """
        Performs batch inference on images.
        @param img_list: List of images.
        @param batch_size: Size of each batch.
        @param score_threshold: Threshold.
        @return: List with predicted class and score.
        """
        batch_holder = np.zeros((batch_size, 250, 250, 3))
        result_list = []

        # Preprocess data
        for index, tree_img in enumerate(img_list):
            tree_tensor = self.preprocess_img(tree_img)
            batch_holder[index] = tree_tensor

        start = time.perf_counter()
        predictions = self.model.predict_on_batch(batch_holder)
        stop = time.perf_counter() - start

        print("Time taken for batch:", stop, " - ", str(batch_size), " trees. - Per tree:", stop / batch_size)

        for data in predictions:
            values = [data[0].numpy(), data[1].numpy()]
            score = tf.nn.softmax(values)

            prediction = [self.class_names[np.argmax(score)], 100 * np.max(score)]
            result_list.append(prediction)

        return result_list

    def load_model(self, model_path):
        print("Loading classifier model...")
        self.model = keras.models.load_model(model_path)

    def save_model(self):
        self.model.save(filepath=classifier("ep_20_no_r.h5"))

