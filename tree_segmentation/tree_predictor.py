import json

from PIL import Image
from matplotlib import pyplot

from tree_segmentation import parser
from tree_segmentation.deepforest import *
from tree_segmentation.deepforest import deepforest
from tree_segmentation.image import get_gps_data



class TreePredictor:
    """Predicts and store trees as images and bounding box data.
    If you don't load a model, the default one will be automatically loaded.
    All training will effect the currently loaded model.
    """

    def __init__(self):
        self.current_model = deepforest.deepforest()
        self.current_model.use_release()

    def load_model(self, model):
        """Load another model to be used for prediction/training.
        @param model: Model to use."""
        self.current_model = deepforest.deepforest(saved_model=get_model(model))

    def predict_and_show(self, image):
        """Predict and show all identified trees in an pyplot-window."""
        image_path = get_data(image)
        image = self.current_model.predict_image(image_path=image_path, return_plot=True)
        pyplot.imshow(image[:, :, ::-1])
        pyplot.show()

    def predict_and_store_trees_from_image(self,
                                           image_path,
                                           save_folder=None,
                                           score_threshold=0,
                                           tree_count_threshold=None,
                                           start_count_from=None,
                                           save_json=False) -> int:
        """
        Predict and then store the results as images.
        Saves all the bounding box data in a json-file.

        @param image_path: Forest image.
        @param save_folder: The folder to store all identified trees.
        @param score_threshold: Threshold for tree accuracy. 0.0-1.0
        @param tree_count_threshold: Set a limit for number of trees to be saved.
        @param start_count_from: Name images starting from this number: tree_33 for example.
        @return: Number of images saved.
        @param save_json: If you want to save the bounding box data for each tree.
        """

        image_name = os.path.basename(image_path)

        # Create save folder
        if save_folder is None:
            save_folder = os.path.join(get_output_dir(), image_path.split(sep=".")[0])

            if not os.path.exists(save_folder):
                try:
                    os.makedirs(save_folder)
                except IOError as e:
                    print(e)
                    return
                else:
                    print("Created folder:", save_folder)

        print("Save folder:", save_folder)
        print("Predicting trees...")

        try:
            image_path = get_data(image_path)
            bounding_boxes = self.current_model.predict_image(image_path, return_plot=False)
            tree_count = len(bounding_boxes)

            if tree_count_threshold is not None:
                if tree_count < tree_count_threshold:
                    print("Trees found:", tree_count)
                    print("Tree count is lower than the threshold")
                    return
            elif tree_count == 0:
                print("No trees found")
                return 0

            json_data = None
            tree_img = Image.open(image_path)

            if save_json is True:
                json_data = {"image": image_name, "latitude": [], "longitude": [], "altitude": 0.0, "trees": []}
                gps = get_gps_data(image_path)
                json_data["latitude"] = list(gps["latitude"])
                json_data["longitude"] = list(gps["longitude"])
                json_data["altitude"] = gps["altitude"]

            picture_count = 0
            saved_images = 0

            if start_count_from is not None:
                picture_count = start_count_from

            for index, row in bounding_boxes.iterrows():

                img_name = "tree_" + str(picture_count)
                x_min = round(row["xmin"])
                y_min = round(row["ymin"])
                x_max = round(row["xmax"])
                y_max = round(row["ymax"])
                score = round(row["score"], 2)

                if score < score_threshold:
                    continue

                if save_json is True:
                    json_data["trees"].append({
                        "tree_name": img_name,
                        "x_min": + x_min,
                        "x_max": + x_max,
                        "y_min": + y_min,
                        "y_max": + y_max,
                        "accuracy": + score,
                    })

                cropped_img = tree_img.crop((x_min, y_min, x_max, y_max))
                cropped_img.save(os.path.join(save_folder, img_name + ".png"))
                saved_images += 1
                picture_count += 1

        except IOError as e:
            print(e)
            return 0
        else:
            if save_json is True:
                with open(os.path.join(save_folder, "json.txt"), "w") as json_file:
                    json_str = (str(json_data)).replace("'", '"')
                    json_data = json.loads(json_str)
                    json.dump(json_data, json_file, indent=4)

        print("Pictures saved:", saved_images)
        return saved_images

    def predict_and_store_trees_from_folder(self, folder_name, save_folder, score_threshold=0, save_json=True):
        """
        @param save_json: Whether to save bounding box data in a json file.
        @param folder_name: Folder of the images to perform prediction.
        @param save_folder: Save location for all image the result folders.
        @param score_threshold: Threshold for tree accuracy. 0.0-1.0
        """
        save_folder = get_output(save_folder)

        # Create folder to save result
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        picture_count = 0

        folder_path = get_data(folder_name)
        images = len(os.listdir(folder_path))

        # Go through all the pictures and save the results
        for index, image in enumerate(os.listdir(folder_path), start=1):
            print("Image ", index, " of ", images)

            picture_save_folder = os.path.join(get_output_dir(), image.split(".")[0])
            try:
                os.mkdir(picture_save_folder)
            except Exception as e:
                print(e)
            else:
                picture_path = os.path.join(folder_path, image)
                picture_count += self.predict_and_store_trees_from_image(picture_path,
                                                                         save_folder=picture_save_folder,
                                                                         score_threshold=score_threshold,
                                                                         start_count_from=picture_count,
                                                                         save_json=save_json)

    def evaluate_model(self, csv_file):
        """
        Evaluates the models' accuracy and prints the mean average precision.
        @param csv_file: CSV-file containing annotation data for a specific image in the deepforest/data folder.
        """
        annotations_file = get_data(csv_file)
        eval_data = self.current_model.evaluate_generator(annotations=annotations_file)
        print("Mean Average Precision is: {:.3f}".format(eval_data))

    def train_model(self):
        """
        Trains the current loaded model. You need to save the model to save the progress.
        The data set to train on should be placed in the model/train directory.
        """
        self.current_model.config["epochs"] = 30
        self.current_model.config["save-snapshot"] = False
        self.current_model.config["steps"] = 1

        for file in get_train_dir():

            file_name, file_extension = os.path.splitext(file)

            if file_extension == ".json":
                json_file_path = get_train(file)
                csv_file = parser.read_and_parse_json(json_file_path, get_train_dir())

                print("Starting training on ", file)
                self.current_model.train(annotations=csv_file, input_type="fit_generator")

    def save_model(self, model_name):
        """Saves the current loaded model in the model directory."""
        self.current_model.model.save(os.path.join(get_model_dir(), model_name + ".h5"))
