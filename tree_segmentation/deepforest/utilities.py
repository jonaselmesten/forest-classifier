# utility functions for demo
import csv
import json
import os
import urllib
import warnings

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

with warnings.catch_warnings():
    # Suppress some of the verbose tensorboard warnings,
    # compromise to avoid numpy version errors
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

from tree_segmentation.keras_retinanet import models
from keras.utils import multi_gpu_model

from tree_segmentation.deepforest import _ROOT, get_model


def label_to_name(class_dict, label):
    """Map label to name."""
    name = class_dict[label]
    return name


def read_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

    return config


def read_model(model_path):
    """Read keras retinanet model from keras.model.save()"""
    # Suppress user warning, module does not need to be compiled for prediction
    with warnings.catch_warnings():
        # warnings.simplefilter('ignore', UserWarning)
        model = models.load_model(model_path, backbone_name='resnet50')

    return model


class DownloadProgressBar(tqdm):
    """Download progress bar class."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def round_with_floats(x):
    """Check if string x is float or int, return int, rounded if needed."""

    try:
        result = int(x)
    except:
        warnings.warn(
            "Annotations files contained non-integer coordinates. "
            "These coordinates were rounded to nearest int. "
            "All coordinates must correspond to pixels in the image coordinate system. "
            "If you are attempting to use projected data, "
            "first convert it into image coordinates see FAQ for suggestions.")
        result = int(np.round(float(x)))

    return result


def update_labels(annotations_file, training_file, current_labels):
    """Create a label list in the format accepted by keras retinanet.

    Args:
        annotations_file : an annotation csv in the retinanet
        format path/to/image.png,x1,y1,x2,y2,class_name

    Returns:
    """

    annotations = pd.read_csv(annotations_file, names=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    # Get unique labels
    labels = annotations.label.dropna().unique()
    n_classes = labels.shape[0]
    size = len(current_labels)

    print("There are {} unique labels: {} ".format(n_classes, list(labels)))

    # Write labels
    with open(training_file, 'a+', newline='\n', encoding='utf-8') as csv_file:

        writer = csv.writer(csv_file)

        for index, label in enumerate(labels):
            if label not in current_labels.values():
                writer.writerow([label, size])
                current_labels[size] = label
                size += 1
                print(index, label)

    return training_file


def number_of_images(annotations_file):
    """How many images in the annotations files?

    Args:
        annotations_file (str):

    Returns:
        n (int): Number of images
    """

    df = pd.read_csv(annotations_file,
                     index_col=False,
                     names=["image_path", "xmin", "ymin", "xmax", "ymax"])
    n = len(df.image_path.unique())
    return n


def format_args(annotations_file, config, images_per_epoch=None):
    """Format config files to match argparse list for retinainet.

    Args:
        annotations_file: a path to a csv
        config (dict): a dictionary object to convert into a list for argparse
        images_per_epoch (int): Override default steps per epoch
            (n images/batch size) by manually setting a number of images

    Returns:
        arg_list (list): a list structure that mimics
            argparse input arguments for retinanet
    """
    # Format args. Retinanet uses argparse, so they need to be passed as a list
    args = {}

    # remember that .yml reads None as a str
    if not config["weights"] == 'None':
        args["--weights"] = config["weights"]

    args["--backbone"] = config["backbone"]
    args["--image-min-side"] = config["image-min-side"]
    args["--multi-gpu"] = config["multi-gpu"]
    args["--epochs"] = config["epochs"]
    if images_per_epoch:
        args["--steps"] = round(images_per_epoch / int(config["batch_size"]))
    else:
        args["--steps"] = round(
            int(number_of_images(annotations_file)) / int(config["batch_size"]))

    args["--batch-size"] = config["batch_size"]
    args["--tensorboard-dir"] = None
    args["--workers"] = config["workers"]
    args["--max-queue-size"] = config["max_queue_size"]
    args["--freeze-layers"] = config["freeze_layers"]
    args["--score-threshold"] = config["score_threshold"]

    if config["save_path"]:
        args["--save-path"] = config["save_path"]

    if config["snapshot_path"]:
        args["--snapshot-path"] = config["snapshot_path"]

    arg_list = [[k, v] for k, v in args.items()]
    arg_list = [val for sublist in arg_list for val in sublist]

    # boolean arguments
    if config["save-snapshot"] is False:
        print("Disabling snapshot saving")
        arg_list = arg_list + ["--no-snapshots"]

    if config["freeze_resnet"] is True:
        arg_list = arg_list + ["--freeze-backbone"]

    if config["random_transform"] is True:
        print("Turning on random transform generator")
        arg_list = arg_list + ["--random-transform"]

    if config["multi-gpu"] > 1:
        arg_list = arg_list + ["--multi-gpu-force"]

    if config["multiprocessing"]:
        arg_list = arg_list + ["--multiprocessing"]

    # positional arguments first
    arg_list = arg_list + ["csv", annotations_file]

    if not config["validation_annotations"] == "None":
        arg_list = arg_list + ["--val-annotations", config["validation_annotations"]]

    # All need to be str classes to mimic sys.arg
    arg_list = [str(x) for x in arg_list]

    return arg_list
