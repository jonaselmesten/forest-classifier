import csv
import os
import random

import keras_preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator


def visualize_data_transformation(data_generator):
    print(data_generator)

    augmented_images = [data_generator[0][0][0] for i in range(9)]

    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    axes = axes.flatten()
    for img, ax in zip(augmented_images, axes):
        ax.imshow(img)
    plt.show()


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
