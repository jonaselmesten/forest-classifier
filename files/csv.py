import csv
import os
import pathlib

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
