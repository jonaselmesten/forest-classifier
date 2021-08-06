import os
import sys
import time
from collections import deque
from glob import glob
from shutil import copy2
from tkinter import filedialog, Tk

import eel

# from classifier.tree_classifier import extract_class_trees, TreeClassifier
from classifier.augmentation import extract_class_trees
from files.csv import overwrite_csv_file, save_csv_file
from folders import gui, tree_seg_train, classifier, tree_seg_model
from gui.menu import MainWin
from gui.window import TreeWin
from segmentation.segment import BoundingBoxPredictor


class Application:

    def __init__(self):

        eel.init("gui")

        self.classifier, self.predictor = self.load_models()
        self.main_window = self.create_main_window()
        self.main_window.mainloop()

        print("MENU OPEN")

    def load_models(self):

        print("Loading models...")

        # classifier = TreeClassifier()
        # classifier.load_model(get_c("ep_20_no_r.h5"))
        classifier = None
        predictor = BoundingBoxPredictor()

        return classifier, predictor

    def create_main_window(self):
        return MainWin(run_prediction=self.run_prediction,
                       add_images=self.copy_images,
                       annotate_prediction=self.open_annotation_window)

    def copy_images(self):
        """
        Opens a file dialog to select images to be copied.
        """
        root = Tk()
        root.title("Select training images")
        root.withdraw()
        root.attributes("-topmost", True)
        selected_files = filedialog.askopenfilenames(parent=root)
        for file in selected_files:
            copy2(file, gui())

    def get_remaining_time(self, time_taken, images_left):
        """
        Creates an estimated of the time left for the remaining images.
        @param time_taken: List of time samples.
        @param images_left: Total images left.
        @return: Estimate of remaining time.
        """
        if len(time_taken) == 0:
            return "∞"

        total_sec = (sum(time_taken) / len(time_taken)) * images_left

        remaining_sec = round(total_sec % 60)
        remaining_min = int(total_sec // 60)

        return str(remaining_min) + ":" + str(remaining_sec)

    def run_prediction(self, bounding_box_threshold=0.40):
        """
        Runs the tree bounding box & species prediction.
        Will also display the estimated time left if there is a big amount of images to predict on.

        For each tree: [(x, y, x, y), Class, Score]
        """

        images = []
        time_taken = deque(maxlen=10)
        start_gpu = time.perf_counter()

        # Gather all images
        for ext in ("*.gif", "*.png", "*.jpg"):
            images.extend(glob(os.path.join(gui(), ext)))

        image_total = len(images)
        image_count = len(images)
        inc_value = 100 / image_count

        tree_data = dict()

        # Predict & classify each image.
        for index, image_path in enumerate(images, start=1):

            start_time = time.perf_counter()

            csv_path = image_path.split(sep=".")[0] + ".csv"
            tree_data[image_path] = list()

            # Image has already been predicted on earlier.
            if os.path.exists(csv_path):
                print("Already exists:", os.path.basename(csv_path))
                continue

            # Update time left.
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

            # Predict tree bounding boxes.
            img_list, bb_list = self.predictor.predict_trees(image_path=image_path,
                                                             score_threshold=bounding_box_threshold)
            result_list = []

            # TODO: Make classifier work.
            for i in range(len(img_list)):
                result_list.append(["Tree", 0.73] * len(img_list))
            print(result_list)
            # result_list = self.classifier.classify_batch_of_trees(img_list, batch_size=len(img_list))
            # TODO: --------------------

            # Start assembling final data list.
            for i, coordinate in enumerate(bb_list):
                tree_data[image_path].append(list(coordinate))
                tree_data[image_path][i].append(result_list[i][0])
                tree_data[image_path][i].append(result_list[i][1])

            stop_time = time.perf_counter() - start_time
            time_taken.append(stop_time)

            image_count -= 1

        print("Time per img:", (time.perf_counter() - start_gpu) / len(images))
        print("Size of result in kB:", sys.getsizeof(tree_data) / 1000)

        self.main_window.destroy()

        # Let user correct predicted trees in each image
        self.show_tree_class_prediction(tree_data)

        self.create_main_window()

    def show_tree_class_prediction(self, tree_data, accuracy_threshold=0.8):
        """
        Opens up a window for the user to remove trees that was wrongfully predicted.
        This makes the last modifications in the via.html-view easier and it helps with
        showing an overview how well the current model can predict.

        For each tree: [(x, y, x, y), Class, Score]

        @param tree_data: Tree species prediction data.
        @param accuracy_threshold: Trees with prediction above this threshold won't be shown.
        """
        # TODO: Jump over empty images
        # TODO: Continue on one page per sort: 10 images all the time, show total of left for species.

        tree_species = {}

        # Group up trees according to species.
        for image_path, tree_list in tree_data.items():

            # Goes through all classes and images.
            for tree in sorted(tree_list, key=lambda i: i[5]):

                class_name = tree[4]

                if class_name not in tree_species:
                    tree_species[class_name] = list()

                accuracy = tree[5]

                # Continue if the prediction was above the threshold.
                if accuracy >= accuracy_threshold:
                    continue

                tree_species[class_name].append([image_path, tree])

        # Open a new window for each species for the user to go through.
        for class_name, trees in tree_species.items():
            tree_window = TreeWin(class_name, trees)
            tree_window.mainloop()

        # Save all prediction data as csv.
        for image_path, tree_list in tree_data.items():
            save_csv_file(image_path, gui(), tree_list)

    def open_annotation_window(self):
        eel.start("via.html", mode="mozilla")


app = Application()
