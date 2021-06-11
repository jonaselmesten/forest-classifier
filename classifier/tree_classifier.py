import csv
import os
import pathlib
import random
import time
import timeit
from builtins import sum

import keras_preprocessing
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


from classifier.layers import Rescaling
from folders import get_ct, get_c


def create_aug_trees(pil_img):
    img_arr = keras_preprocessing.image.img_to_array(pil_img)

    pil_img_list = []
    generator = ImageDataGenerator()

    for i in range(9):
        test = generator.apply_transform(img_arr, transform_parameters={
            "theta": random.randrange(-10, 10),
            "shear": random.randrange(0, 20),
            "zx": random.uniform(0.9, 1.1),
            "zy": random.uniform(0.9, 1.1),
            "flip_horizontal": True if random.random() > 0.5 else False,
            "flip_vertical": True if random.random() > 0.5 else False,
            "brightness": random.uniform(0.9, 1.1)
        })
        final = keras_preprocessing.image.array_to_img(test)
        pil_img_list.append(final)

    return pil_img_list


def extract_class_trees(image_file, csv_file):
    image = Image.open(image_file)

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

    with open(csv_file, "r", newline="\n", encoding="utf-8")as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            if row[5].capitalize().strip() != "Tree":

                class_name = row[5].lower()

                # Create class folder
                if not os.path.exists(get_ct(class_name)):
                    os.makedirs(get_ct(class_name))

                pixels = tuple(map(int, [row[1], row[2], row[3], row[4]]))
                image_count = len(os.listdir(get_ct(class_name)))

                # Save original image
                img_name = class_name + "\\" + class_name.lower() + "_" + str(image_count)
                org_image = image.crop(pixels)
                org_image.save(get_ct(img_name + ".jpg"), "JPEG", quality=95)

                img_list = create_aug_trees(org_image)

                for aug_img in img_list:
                    image_count += 1
                    img_name = class_name + "\\" + class_name.lower() + "_" + str(image_count)
                    aug_img.save(get_ct(img_name + ".jpg"), "JPEG", quality=95)


def test_accuracy(folder_path, class_name, classifier_obj):
    trees = [os.path.join(r, file) for r, d, f in os.walk(folder_path) for file in f]

    tree_count = len(trees)
    error_count = 0
    under_80_count = 0
    under_70_count = 0
    under_60_count = 0
    under_50_count = 0
    lowest_acc = 100.0

    print("Classifying ", len(trees), " trees...")
    print("Class:", class_name)

    start_time = timeit.default_timer()

    for tree in trees:
        predicted_class, accuracy = classifier_obj.classify_single_tree(tree)

        if accuracy < lowest_acc:
            lowest_acc = accuracy

        if accuracy < 80.0:
            under_80_count += 1
        if accuracy < 70.0:
            under_70_count += 1
        if accuracy < 60.0:
            under_60_count += 1
        if accuracy < 50.0:
            under_50_count += 1

        if predicted_class.lower() != class_name.lower():
            error_count += 1

    elapsed = timeit.default_timer() - start_time
    print("Took {time} seconds to classify each tree.".format(time=elapsed / len(trees)))
    print("Took {time} seconds to classify all trees.".format(time=elapsed))
    print("--------------------------")
    print("Lowest accuracy:", lowest_acc)
    print("Trees under 80.0 accuracy:", under_80_count)
    print("Trees under 70.0 accuracy:", under_70_count)
    print("Trees under 60.0 accuracy:", under_60_count)
    print("Trees under 50.0 accuracy:", under_50_count)
    print(error_count, " out of ", tree_count, " was predicted wrong.")
    print("--------------------------")


def visualize_data_transformation(data_generator):
    print(data_generator)

    augmented_images = [data_generator[0][0][0] for i in range(9)]

    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    axes = axes.flatten()
    for img, ax in zip(augmented_images, axes):
        ax.imshow(img)
    plt.show()


class TreeClassifier:

    # TODO: Add config files and reader just like in deepforest.
    def __init__(self, saved_model=None):
        self.history = None
        self.rescale = Rescaling(1. / 255)
        self.data_dir = get_ct()
        self.data_dir = pathlib.Path(self.data_dir)
        self.classes = len(next(os.walk(self.data_dir))[1])
        self.batch_size = 200
        self.img_size = (250, 250)
        self.class_names = ["Birch", "Spruce"]
        print("Initializing tree classifier...")
        print("Classes:", self.class_names)

        # Sequential models without an `input_shape`
        # passed to the first layer cannot reload their optimizer state.

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

    def train(self):

        epochs = 10
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

        # Preprocess
        resize_img = pil_img.resize(self.img_size, Image.NEAREST)
        img_array = keras.preprocessing.image.img_to_array(resize_img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = self.rescale(img_array)

        return img_array

    def classify_batch_of_trees(self, img_list, batch_size=400, score_threshold=0.70):
        # TODO: Raise EXC when img and batch size is different
        batch_holder = np.zeros((batch_size, 250, 250, 3))
        result_list = []

        # Preprocess data
        for index, tree_img in enumerate(img_list):
            tree_tensor = self.preprocess_img(tree_img)
            batch_holder[index] = tree_tensor

        # print("Type:", type(img_list[0]))
        # print("Batch holder size:", len(batch_holder))
        # print("Img holder size:", len(img_list))

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
        self.model.save(filepath=get_c("ep_20_no_r.h5"))

    def test_speed(self):

        # Avg time: 0.029

        dir = get_ct("spruce")
        time_list = []
        count = 0
        pre_imgs = []

        batch_size = 400
        batch_holder = np.zeros((400, 250, 250, 3))

        normal_pred = []
        batch_pred = []

        # Preprocess data
        for i, img in enumerate(pathlib.Path(dir).glob("*.jpg")):

            if i == batch_size:
                break

            pil_img = Image.open(img)
            tree_img = pil_img.resize(self.img_size, Image.NEAREST)
            img_array = keras.preprocessing.image.img_to_array(tree_img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = self.rescale(img_array)

            start = time.perf_counter()
            a, b = self.classify_single_tree(tree_img)
            stop = time.perf_counter() - start
            time_list.append(stop)

            normal_pred.append(str(a) + str(round(b, 3)))

            batch_holder[i] = img_array

        start = time.perf_counter()
        finished = self.model.predict_on_batch(batch_holder)
        stop = time.perf_counter() - start

        print("NORMAL - Total:", sum(time_list))
        print("NORMAL - per tree:", sum(time_list) / batch_size)
        print("BATCH - Total:", stop)
        print("BATCH - per tree:", stop / batch_size)
        # Lowest 400
        # 0.006692224499999999

        for data in finished:
            values = [data[0].numpy(), data[1].numpy()]
            score = tf.nn.softmax(values)

            a, b = self.class_names[np.argmax(score)], 100 * np.max(score)
            batch_pred.append(str(a) + str(round(b, 3)))

        for i in range(len(batch_pred)):

            if batch_pred[i] != normal_pred[i]:
                print("ERROR:", batch_pred[i], normal_pred[i])

    def warm_up(self):

        test_list = []
        size = 1

        for i in range(size):
            img = get_ct("spruce\\spruce_0.jpg")
            test_list.append(Image.open(img))

        result_list = self.classify_batch_of_trees(test_list, batch_size=size)

        print("Classifier batch - OK")


# Rescaling kan inte sparas i save model....
#classifier = TreeClassifier()
#classifier.load_model(get_c("ep_20_no_r.h5"))


# classifier.train()
# classifier.plot_history()
# classifier.save_model()

# test_accuracy(get_ct("test"), "spruce", tc)
