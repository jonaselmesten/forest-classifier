import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_cmd(file_name=None) -> os.path:
    """Returns cluster model folder or a file in that folder if you specify a file name"""
    if file_name:
        return os.path.join(_ROOT, "clustering\\model", file_name)
    else:
        return os.path.join(_ROOT, "clustering\\model")


def get_tso(file_name=None) -> os.path:
    """Returns tree segmentation output folder  or a file in that folder if you specify a file name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\output", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\output")


def get_tsd(file_name=None) -> os.path:
    """Returns tree segmentation data folder or a file in that folder if you specify a file name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\data", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\data")
