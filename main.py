import os
import os
import sys
import time
from collections import deque
from glob import glob
from shutil import copy2
from tkinter import filedialog, Tk

import eel

from classifier.tree_classifier import extract_class_trees, TreeClassifier
from files.csv import update_csv_file, save_csv_file
from folders import get_gui, get_tsmt, get_c, get_tsm
from gui.menu import MainMenu
from gui.window import TreeWindow
from tree_segmentation.image import flip_img_csv
from tree_segmentation.segment import TreePredictor


@eel.expose
def send_to_training(file, csv_data, train_both):
    if file is None:
        return

    image_file = get_gui(file)
    image_name = os.path.basename(image_file)

    csv_file = get_gui(file.split(sep=".")[0] + ".csv")
    csv_name = os.path.basename(csv_file)

    update_csv_file(csv_file, csv_data)

    if os.path.isfile(image_file) and os.path.isfile(csv_file):

        # Move to segmentation training
        if train_both:
            extract_class_trees(image_file, csv_file)
            flip_img_csv(image_file, csv_file, get_tsmt())
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
    os.remove(get_gui(image))
    os.remove(get_gui(csv_file))


class Application:

    def __init__(self):

        eel.init('gui')

        self.classifier, self.predictor = self.load_models()
        self.main_window = self.create_main_window()
        self.main_window.mainloop()

        print("MENU OPEN")

    def load_models(self):

        print("Loading models...")

        classifier = TreeClassifier()
        classifier.load_model(get_c("ep_20_no_r.h5"))
        predictor = TreePredictor()

        return classifier, predictor

    def create_main_window(self):
        return MainMenu(run_prediction=self.run_prediction,
                        add_images=self.copy_images,
                        annotate_prediction=self.open_annotation)

    def copy_images(self):
        root = Tk()
        root.title("Select training images")
        root.withdraw()
        root.attributes("-topmost", True)
        selected_files = filedialog.askopenfilenames(parent=root)
        for file in selected_files:
            copy2(file, get_gui())

    def get_remaining_time(self, time_taken, images_left):

        if len(time_taken) == 0:
            return "∞"

        total_sec = (sum(time_taken) / len(time_taken)) * images_left

        remaining_sec = round(total_sec % 60)
        remaining_min = int(total_sec // 60)

        return str(remaining_min) + ":" + str(remaining_sec)

    # TODO: Create application logic module to simplify
    def run_prediction(self):

        images = []
        time_taken = deque(maxlen=10)
        start_gpu = time.perf_counter()

        # Gather all images
        for ext in ("*.gif", "*.png", "*.jpg"):
            images.extend(glob(os.path.join(get_gui(), ext)))

        image_total = len(images)
        image_count = len(images)
        inc_value = 100 / image_count

        tree_data = dict()

        # Predict & classify each image
        for index, image_path in enumerate(images, start=1):

            start_time = time.perf_counter()

            csv_path = image_path.split(sep=".")[0] + ".csv"
            tree_data[image_path] = list()

            # TODO: How does this effect prog. bar?
            if os.path.exists(csv_path):
                print("Already exists:", os.path.basename(csv_path))
                continue

            # Update GUI
            if index == 1:
                remaining_time = self.get_remaining_time(time_taken, image_count)
            elif index > 6:
                if index % 3 == 0:
                    remaining_time = self.get_remaining_time(time_taken, image_count)

            self.main_window.start_progress_bar()
            self.main_window.update_text("Predicting trees on image "
                                         + str(index) + " out of "
                                         + str(image_total) + " Time left:"
                                         + remaining_time)
            self.main_window.inc_progress_bar(inc_value)

            # Predict & classify
            img_list, bb_list = self.predictor.predict_trees(image_path=image_path, score_threshold=0.40)
            #result_list = self.classifier.classify_batch_of_trees(img_list, batch_size=len(img_list))

            # Start assembling final data list
            for i, coordinate in enumerate(bb_list):
                tree_data[image_path].append(list(coordinate))
                #tree_data[image_path][i].append(result_list[i][0])
                #tree_data[image_path][i].append(result_list[i][1])

                # For each tree: [(x, y, x, y), Class, Score]

            stop_time = time.perf_counter() - start_time
            time_taken.append(stop_time)

            image_count -= 1

        print("Time per img:", (time.perf_counter() - start_gpu) / len(images))
        print("Size of result in kB:", sys.getsizeof(tree_data) / 1000)

        self.main_window.destroy()

        # Let user correct predicted trees in each image
        self.show_tree_class_prediction(tree_data)

        self.create_main_window()

    def show_tree_class_prediction(self, tree_data):

        # TODO: Jump over empty images
        # TODO: Continue on one page per sort: 10 images all the time, show total of left for spieces.

        tree_species = {}

        # Group up trees according to species
        for image_path, tree_list in tree_data.items():

            # Goes through all classes and images
            for tree in sorted(tree_list, key=lambda i: i[5]):

                class_name = tree[4]

                if class_name not in tree_species:
                    tree_species[class_name] = list()

                accuracy = tree[5]

                if accuracy > 80.0:
                    continue

                tree_species[class_name].append([image_path, tree])

        # Open a new window for each species
        for class_name, trees in tree_species.items():
            tree_window = TreeWindow(class_name, trees)
            tree_window.mainloop()

        print(tree_data)

        # Save all prediction data as csv
        for image_path, tree_list in tree_data.items():
            save_csv_file(image_path, get_gui(), tree_list)

    # TODO: Ladda alla bilder här
    def open_annotation(self):
        eel.start(get_gui("via.html"))


app = Application()
