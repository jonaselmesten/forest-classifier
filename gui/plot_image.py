import json
import os

from matplotlib.image import imread
from pylab import plt
from folders import get_tso


def plot_trees_and_select(tree_folder) -> int:
    """
    Plots trees of the highest accuracy from a folder of images.
    Then asks for selection through the terminal.

    :param tree_folder: Folder that contains tree images and json
    :return: Numbers for: tree1-img1, tree1-img2, tree2-img1, tree2-img1
    """
    rows = 5
    columns = 5
    trees = set()

    fig, image_grid = plt.subplots(rows, columns)

    # Open the folder and use the json-file to extract trees
    with open(os.path.join(tree_folder, "json.txt")) as json_file:
        data = json.load(json_file)

        row = 0
        column = 0

        # Plot each tree
        for count, tree in enumerate(data["trees"]):

            tree_name = tree["tree_name"]
            image = imread(get_tso(tree_name + ".png"))
            image_grid[row, column].imshow(image)
            image_grid[row, column].title.set_text(tree["tree_name"])
            image_grid[row, column].set_yticks([])
            image_grid[row, column].set_xticks([])

            tree_num = int("".join(filter(str.isdigit, tree_name)))
            trees.add(tree_num)

            column += 1

            if column == 5:
                row += 1
                column = 0

            if row == 5:
                break

        fig.set_size_inches(8, 8)
        fig.suptitle("Choose two different trees. Write in terminal. Ex: 2 14")
        plt.show()

    # Get tree 1
    while True:
        tree_x_1, tree_x_2 = [int(x) for x in input("First tree - Enter two numbers:").split()]

        if tree_x_1 in trees and tree_x_2 in trees and tree_x_1 != tree_x_2:
            break
        else:
            print("Value error - Try again")

    # Get tree 2
    while True:
        tree_y_1, tree_y_2 = [int(x) for x in input("Second tree - Enter two numbers:").split()]

        if tree_y_1 in trees and tree_y_2 in trees and tree_y_1 != tree_y_2:
            break
        else:
            print("Value error - Try again")

    return tree_x_1, tree_x_2, tree_y_1, tree_y_2
