import os
import pathlib

import PIL
from PIL.Image import Image

import csv
from folders import gui


def save_csv_file(image_path, csv_save_folder, tree_list):
    file_name = os.path.basename(image_path)
    file_name_csv = file_name.split(sep=".")[0] + ".csv"

    print("Saving predicted trees:", file_name_csv)

    with open(os.path.join(csv_save_folder, file_name_csv), "w", newline="\n", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator='\n')

        for tree in tree_list:
            csv_writer.writerow([file_name, tree[0], tree[1], tree[2], tree[3], tree[4]])


def merge_csv_files(folder_path, file_name):
    with open(os.path.join(folder_path, file_name + ".csv"), "w", newline="\n", encoding="utf-8") as save_file:

        for file in pathlib.Path(folder_path).glob("**/*.csv"):

            current_file = os.path.basename(file).split(sep=".")[0]

            if current_file == file_name:
                continue

            with open(file, 'r') as csv_file:

                for line in csv_file:
                    save_file.write(line)


def update_csv_file(file_name, csv_data):
    file_path = gui(file_name)

    with open(file_path, "w", newline="\n", encoding="utf-8")as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator='\n')

        for row in csv_data:
            data = row.split(",")
            csv_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5]])


def flip_img_csv(image_path, csv_path, save_folder):
    """

    @param image_path:
    @param csv_path:
    @param save_folder:
    """
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
