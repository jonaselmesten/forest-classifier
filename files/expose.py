import os
import shutil

import eel

from classifier.augmentation import extract_class_trees
from files.csv import overwrite_csv_file, flip_img_csv
from folders import tree_seg_train, gui


@eel.expose
def send_to_training(file, csv_data, train_both):
    """
    Method to be used after all the desired changes has been made to a predict image.
    Suitable annotation data for classifier & bounding box model will be created.

    @param file: File name for both image and csv.
    @param csv_data: Csv-file to be overwritten with adjusted data.
    @param train_both: True - creates annotation data for both classifier and bounding box model.
    False - just for classifier.
    @return:
    """
    if file is None:
        return

    image_file = gui(file)
    image_name = os.path.basename(image_file)

    csv_file = gui(file.split(sep=".")[0] + ".csv")
    csv_name = os.path.basename(csv_file)

    overwrite_csv_file(csv_file, csv_data)

    if os.path.isfile(image_file) and os.path.isfile(csv_file):

        if train_both:
            extract_class_trees(image_file, csv_file)
            flip_img_csv(image_file, csv_file, tree_seg_train())

            shutil.move(image_file, tree_seg_train(image_name))
            shutil.move(csv_file, tree_seg_train(csv_name))
            remove_file(image_file)
        else:

            extract_class_trees(image_file, csv_file)
            eel.project_file_remove_confirmed()
            remove_file(image_file)

    else:
        raise FileNotFoundError("Couldn't find the file:", file)


@eel.expose
def remove_file(file_name):
    """
    Removes image & csv-file from the gui-folder.
    @param file_name: File to be removed.
    """
    image = file_name
    csv_file = file_name.split(sep=".")[0] + ".csv"

    try:
        os.remove(gui(image))
        os.remove(gui(csv_file))
    except IOError as e:
        print(e)
        print("Failed to remove file:", file_name)

