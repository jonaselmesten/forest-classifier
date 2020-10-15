import json
import csv
import os

from tree_segmentation.deepforest import get_data


def read_and_parse_json(json_file: os.path, save_folder: os.path) -> os.path:
    """
    Reads and parses a json-file to the correct format for training/evaluation.
    @param save_folder: Folder to save csv-file.
    @param json_file: Text file containing json.
    @return: Path to csv file - None if IO-error.
    """
    json_file = get_data(json_file)

    file = open(json_file)
    json_dict = json.load(file)
    value = next(iter(json_dict.values()))

    image_path = value["filename"]

    # Create csv file
    csv_file = image_path.split(".")
    csv_file[1] = "csv"
    csv_file = ".".join(csv_file)
    csv_file_path = os.path.join(save_folder, csv_file)

    try:
        csv_file = open(csv_file_path, 'w', newline="")
        file_writer = csv.writer(csv_file)

        # Extract all values
        for val in value["regions"]:

            inner = val["shape_attributes"]
            xmin = inner["x"]
            ymin = inner["y"]
            xmax = inner["width"] + xmin
            ymax = inner["height"] + ymin

            if xmin > xmax | ymin > ymax:
                os.remove(csv_file)
                raise ValueError("Xmin/ymin should never be bigger than xmax/ymax, check your file for error.")

            result = [image_path, str(xmin), str(ymin), str(xmax), str(ymax), "Tree"]
            file_writer.writerow(result)

    except (IOError, ValueError) as e:
        print(e)
        return None

    return csv_file_path
