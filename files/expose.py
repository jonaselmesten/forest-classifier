import os

import eel

from classifier.augmentation import extract_class_trees
from files.csv import update_csv_file
from folders import tree_seg_train, gui
from tree_segmentation.image import flip_img_csv


@eel.expose
def send_to_training(file, csv_data, train_both):
    if file is None:
        return

    image_file = gui(file)
    image_name = os.path.basename(image_file)

    csv_file = gui(file.split(sep=".")[0] + ".csv")
    csv_name = os.path.basename(csv_file)

    update_csv_file(csv_file, csv_data)

    if os.path.isfile(image_file) and os.path.isfile(csv_file):

        # Move to segmentation training
        if train_both:
            extract_class_trees(image_file, csv_file)
            flip_img_csv(image_file, csv_file, tree_seg_train())
            #shutil.move(image_file, get_tsmt(image_name))
            #shutil.move(csv_file, get_tsmt(csv_name))
            remove_file(image_file)
        else:
            extract_class_trees(image_file, csv_file)
            eel.project_file_remove_confirmed()
            remove_file(image_file)

    else:
        raise FileNotFoundError


@eel.expose
def remove_file(file_name):
    image = file_name
    csv_file = file_name.split(sep=".")[0] + ".csv"
    os.remove(gui(image))
    os.remove(gui(csv_file))