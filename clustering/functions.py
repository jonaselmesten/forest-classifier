from random import random, randint

import numpy as np

from clustering import model
from clustering.model import Models
from folders import *


def cluster_and_plot(tree_x_1, tree_x_2, tree_y_1, tree_y_2):
    image_folder = get_tso()
    model_path = get_cmd()

    selected_model = Models.INCEPTION_3

    # Prediction and feature is the same size as number of images.
    prediction = model.run_example_INCEPTION_3(image_folder, model_path)

    # TODO: Make size automatic
    # The size differs depending on selected model
    tree_x_arr = np.zeros((2, 1000))
    tree_x_arr[0] = prediction[tree_x_1]
    tree_x_arr[1] = prediction[tree_x_2]

    # The size differs depending on selected model
    tree_y_arr = np.zeros((2, 1000))
    tree_y_arr[0] = prediction[tree_y_1]
    tree_y_arr[1] = prediction[tree_y_2]

    mean2 = np.mean(tree_x_arr, axis=0)
    opposite2 = np.mean(tree_y_arr, axis=0)

    x, img_arr = model.get_nearest_neighbor_and_similarity(prediction, 430, mean2, "mean2", selected_model)
    y, img_arr = model.get_nearest_neighbor_and_similarity(prediction, 430, opposite2, "opposite2", selected_model)

    # K-MEANS
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(pred))
    # y_kmeans = kmeans.predict(np.array(pred))

    # def plot_simpleGraph(X,Y):
    # plt.plot(X, Y,"r.")

    # plt.scatter(simil3,simil4, c=y_kmeans, s=20, cmap='viridis')
    # plt.show

    # PLOT IMAGE INSTEAD OF POINT....
    model.plot_image_cluster(x, y, img_arr, 0.2, image_folder)

    # def getImage(path):
    #    return OffsetImage(plt.imread(path))

    # paths = [
    #    'a.jpg',
    #    'b.jpg',
    #    'c.jpg',
    #    'd.jpg',
    #    'e.jpg']
    # _images = []
    # smil3 = simil3
    # smil4 = simil4
    # x = [0,1,2,3,4]
    # y = [0,1,2,3,4]
