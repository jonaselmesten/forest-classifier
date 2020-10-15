import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("ROOT:", _PROJECT_ROOT)

def get_data(path):
    return os.path.join(_ROOT, '../data', path)


def get_eval(path):
    return os.path.join(_PROJECT_ROOT, "\\model\\eval", path)


def get_train(path):
    return os.path.join(_PROJECT_ROOT, "\\model\\train", path)


def get_train_dir():
    return os.path.join(_PROJECT_ROOT, "\\model\\train")


def get_output(path):
    return os.path.join(_PROJECT_ROOT, "output", path)


def get_output_dir():
    return os.path.join(_PROJECT_ROOT, "output")


def get_model(path):
    return os.path.join(_PROJECT_ROOT, "model", path)


def get_model_dir():
    return os.path.join(_PROJECT_ROOT, "model")
