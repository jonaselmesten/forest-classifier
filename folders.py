import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def tree_seg_output(file_name=None) -> os.path:
    """Returns tree segmentation output folder  or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "segmentation\\output", file_name)
    else:
        return os.path.join(_ROOT, "segmentation\\output")


def tree_seg_data(file_name=None) -> os.path:
    """Returns tree segmentation data folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "segmentation\\data", file_name)
    else:
        return os.path.join(_ROOT, "segmentation\\data")


def tree_seg_train(file_name=None) -> os.path:
    """Returns tree segmentation model train folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "segmentation\\model\\train", file_name)
    else:
        return os.path.join(_ROOT, "segmentation\\model\\train")


def tree_seg_eval(file_name=None) -> os.path:
    """Returns tree segmentation model evaluation folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "segmentation\\model\\eval", file_name)
    else:
        return os.path.join(_ROOT, "segmentation\\model\\eval")


def tree_seg_model(file_name=None) -> os.path:
    """Returns tree segmentation model folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "segmentation\\model", file_name)
    else:
        return os.path.join(_ROOT, "segmentation\\model")


def gui(file_name=None) -> os.path:
    """Returns gui folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "gui", file_name)
    else:
        return os.path.join(_ROOT, "gui")


def classifier_tree(file_name=None) -> os.path:
    """Returns classifier tree folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "classifier\\trees", file_name)
    else:
        return os.path.join(_ROOT, "classifier\\trees")


def classifier(file_name) -> os.path:
    """Returns classifier folder or a file in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "classifier", file_name)
    else:
        return os.path.join(_ROOT, "classifier")