import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_cmd(file_name=None) -> os.path:
    """Returns cluster model folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "clustering\\model", file_name)
    else:
        return os.path.join(_ROOT, "clustering\\model")


def get_co(file_name=None) -> os.path:
    """Returns cluster output folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "clustering\\output", file_name)
    else:
        return os.path.join(_ROOT, "clustering\\output")


def get_tso(file_name=None) -> os.path:
    """Returns tree segmentation output folder  or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\output", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\output")


def get_tsd(file_name=None) -> os.path:
    """Returns tree segmentation data folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\data", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\data")


def get_tsmt(file_name=None) -> os.path:
    """Returns tree segmentation model train folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\model\\train", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\model\\train")


def get_tsme(file_name=None) -> os.path:
    """Returns tree segmentation model evaluation folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\model\\eval", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\model\\eval")


def get_tsm(file_name=None) -> os.path:
    """Returns tree segmentation model folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "tree_segmentation\\model", file_name)
    else:
        return os.path.join(_ROOT, "tree_segmentation\\model")


def get_gui(file_name=None) -> os.path:
    """Returns tree segmentation model folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "gui", file_name)
    else:
        return os.path.join(_ROOT, "gui")


def get_ct(file_name=None) -> os.path:
    """Returns classifier tree folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "classifier\\trees", file_name)
    else:
        return os.path.join(_ROOT, "classifier\\trees")


def get_c(file_name=None) -> os.path:
    """Returns classifier folder or a files in that folder if you specify a files name"""
    if file_name:
        return os.path.join(_ROOT, "classifier", file_name)
    else:
        return os.path.join(_ROOT, "classifier")