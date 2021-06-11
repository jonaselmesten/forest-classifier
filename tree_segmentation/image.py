import csv
import os

import PIL
from PIL import Image
from PIL.ExifTags import TAGS

from files.csv import save_csv_file
from folders import get_gui


def get_gps_data(image_path, tag_id=34853) -> dict:
    """Extracts gps metadata from an image.
    This function is based on DJI FC 6310 camera, it might need some modification
    to be able to extract gps-metadata from images taken with another camera.

    @param image_path: Image to extract from.
    @param tag_id: Tag id for gps-metadata.
    @return: {"latitude": (0,0,0), "longitude": (0,0,0), "altitude": 0.0)} or None if fail.
    """

    try:
        metadata = Image.open(image_path).getexif()
        gps_data = metadata.get(tag_id)
        tag = TAGS.get(tag_id, tag_id)

        if tag != "GPSInfo":
            raise TypeError("GPSInfo was not loaded.", tag, " was loaded.")

    except Exception as e:
        print(e)
        return None
    else:
        return {"latitude": gps_data[2], "longitude": gps_data[4], "altitude": float(gps_data[6])}


def flip_img_csv(image_path, csv_path, save_folder):
    # check both exist
    # remove all if error

    org_img = Image.open(image_path)
    org_name = os.path.basename(image_path).split(sep=".")[0]
    width, height = org_img.size
    tree_list = []

    # Read in original csv data
    with open(csv_path, "r", newline="\n", encoding="utf-8")as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            tree_list.append(row)

    for i in range(1, 4):

        file_path = os.path.join(save_folder, org_name + "_" + str(i))

        # Make to one method
        with open(file_path + ".csv", "w", newline="\n", encoding="utf-8")as csv_file:
            csv_writer = csv.writer(csv_file, lineterminator='\n')

            for row in tree_list:  # x min   y min   x max   y max

                x_min = int(row[1])
                y_min = int(row[2])
                x_max = int(row[3])
                y_max = int(row[4])
                bb_width = x_max - x_min
                bb_height = y_max - y_min

                # TODO: Move this to csv
                if i == 1:
                    x_min = width - x_min - bb_width
                    x_max = x_min + bb_width
                elif i == 2:
                    y_min = height - y_min - bb_height
                    y_max = y_min + bb_height
                    pass
                elif i == 3:
                    x_min = width - x_min - bb_width
                    x_max = x_min + bb_width
                    y_min = height - y_min - bb_height
                    y_max = y_min + bb_height
                elif i == 4:
                    csv_writer.writerow([os.path.basename(image_path), x_min, y_min, x_max, y_max, "Tree"])
                csv_writer.writerow([org_name + "_" + str(i) + ".JPG", x_min, y_min, x_max, y_max, "Tree"])

        if i == 1:
            flip_img = org_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            flip_img.save(file_path + ".jpg", "JPEG", quality=95)
        elif i == 2:
            flip_img = org_img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            flip_img.save(file_path + ".jpg", "JPEG", quality=95)
        elif i == 3:
            flip_img = org_img.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT)
            flip_img.save(file_path + ".jpg", "JPEG", quality=95)
