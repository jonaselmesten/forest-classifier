import enum
import os
import ssl

import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
from keras import models, Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import spatial
from tensorflow.keras.layers import Input
# Cosine similarity: https://medium.com/@salmariazi/computing-image-similarity-with-pre-trained-keras-models-3959d3b94eca


# Current models
class Models(enum.Enum):
    VGG19_MODEL = 1
    INCEPTION_3 = 2
    INCEPTION_RESNET_V2 = 3


def create_model(save_filepath, model_selection=Models.VGG19_MODEL):
    """
    Create a VGG model and save to specified path
    :param save_filepath: Saved model path
    :param model_selection: Model selection from Models-enum
    :return: Created model
    """
    if model_selection == Models.VGG19_MODEL:  # Use VGG19 as base model (Lightest)
        from keras.applications.vgg19 import preprocess_input
        from keras.applications.vgg19 import VGG19
        # loading vgg16 model and using all the layers until the 2 to the    last to use all the learned cnn layers
        ssl._create_default_https_context = ssl._create_unverified_context
        # input_tensor = Input(shape=(224, 224, 3))
        vgg = VGG19(include_top=True)
        base_model = Model(vgg.input, vgg.layers[-2].output)

    if model_selection == Models.INCEPTION_3:  # Use INCEPTION_V3 as base model
        from keras.applications.inception_v3 import preprocess_input
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        # input_tensor = Input(shape=(224, 224, 3))
        input_tensor = Input(shape=(299, 299, 3))
        base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
        # model2 = Model(base_model.input, base_model.layers[-2].output)

    if model_selection == Models.INCEPTION_RESNET_V2:  # Use INCEPTION_RESNET_V2 as base model
        from keras.applications.inception_resnet_v2 import preprocess_input
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        input_tensor = Input(shape=(224, 224, 3))
        # input_tensor = Input(shape=(168, 168, 3))
        base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet',
                                       include_top=False)  # Include Top can only be true if 299x299
        # model2 = Model(base_model.input, base_model.layers[-2].output)

    _selected_model = model_selection
    base_model.save(save_filepath, model_selection)  # saving the model just in case

    return base_model


# Load Images from a folder and convert to an array
def load_images_from_folder(folder, model):

    if not model:
        raise Exception("A model must be selected")

    images = []
    target_size = (299, 299)

    # TODO: Add other target sizes
    if model == Models.INCEPTION_3:
        target_size = (299, 299)
    elif model == Models.INCEPTION_RESNET_V2:
        target_size = (0, 0)
    elif model == Models.VGG19_MODEL:
        target_size = (0, 0)

    print("Target size:", target_size)

    for filename in os.listdir(folder):

        if filename == "json.txt":
            continue

        img = load_img(os.path.join(folder, filename), target_size=target_size)
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
        img = preprocess_input(img)
        img.flatten()

        if img is not None:
            images.append(img)

    return images


# Load a model from specified path
def load_model_from_path(save_filepath, model):
    if model == Models.VGG19_MODEL:
        from keras.applications.vgg19 import preprocess_input
        from keras.applications.vgg19 import VGG19

    elif model == Models.INCEPTION_3:
        from keras.applications.inception_v3 import preprocess_input
        from tensorflow.keras.applications.inception_v3 import InceptionV3

    elif model == Models.INCEPTION_RESNET_V2:
        from keras.applications.inception_resnet_v2 import preprocess_input
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

    model = models.load_model(save_filepath)

    return model


# def get_preds(all_imgs_arr,model):
# getting the extracted features - final shape (number_of_images, 4096)
# preds = model.predict(all_imgs_arr)
# return preds

# get predictions (of object?) and return as array
def get_predictions(images_array, model, model_selection):

    dims = 0
    # Dims will differ depending on model used, VGG: 4096, inception_v3: 1000, Inception_resnet_v2: 25088
    if model_selection == Models.VGG19_MODEL:
        dims = 4096
    elif model_selection == Models.INCEPTION_3:
        dims = 1000
    elif model_selection == Models.INCEPTION_RESNET_V2:
        dims = 25088

    # TODO: Problem if the pictures are bigger than 299x299
    print("Dims:", dims)
    print("Length:", len(images_array))
    print(model)

    all_predictions = np.zeros((len(images_array), dims))

    # Amount of img x dims
    for i in range(np.array(images_array).shape[0]):
        all_predictions[i] = model.predict(images_array[i])

    return all_predictions


# Get features
def get_features(all_imgs_arr, model2):
    # model2 = VGG19(weights='imagenet', include_top=False)

    feature_list = []
    for i in range(np.array(all_imgs_arr).shape[0]):
        # img_data = img_to_array(all_imgs_arr[i])
        # img_data = np.expand_dims(all_imgs_arr[i], axis=0)
        # img_data = preprocess_input(all_imgs_arr[i])
        img_data = all_imgs_arr[i]
        features = np.array(model2.predict(img_data))
        feature_list.append(features.flatten())
    return feature_list


# K = Numer of images to take into account, Master Image is the vector to check similarity with, preds are predictions
def get_nearest_neighbor_and_similarity(predictions, image_count, master_image, save_path, model):

    if model == Models.VGG19_MODEL:
        dims = 4096
    elif model == Models.INCEPTION_3:
        dims = 1000
    elif model == Models.INCEPTION_RESNET_V2:
        dims = 25088

    n_nearest_neighbors = image_count + 1
    trees = 10000
    file_index_to_file_vector = {}

    # Build an index (Approximate Nearest Neighbours)
    t = AnnoyIndex(dims)
    # for i in range(preds.shape[0]):
    i = 0
    j = 0
    for l in predictions:
        file_vector = predictions[i]
        file_index_to_file_vector[i] = file_vector
        t.add_item(i, file_vector)
        i += 1
    t.build(trees)
    t.save(save_path)

    # for i in range(preds.shape[0]):
    for o in predictions:
        master_vector = file_index_to_file_vector[j]
        # Here we assign master vector, SHOULD be one K

        named_nearest_neighbors = []
        similarities = []
        nearest_neighbors = t.get_nns_by_item(j, n_nearest_neighbors)
        j += 1

    # Next we print all the neighbours on one axis, should redo new master and nearest for the second axis to plot
    for j in nearest_neighbors:
        #         print (j)
        neighbor_vector = predictions[j]
        # The distance between objects,/ similarity, cosine for vinkel
        # similarity = 1 - spatial.distance.cosine(master_vector, neighbor_vector)
        similarity = 1 - spatial.distance.cosine(master_image, neighbor_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0
        similarities.append(rounded_similarity)
    return similarities, nearest_neighbors


# NOT CURRENTLY WORKING (Supposed to show similar images)
def get_similar_images(similarities, nearest_neighbors, images1):
    j = 0
    cnt = 0
    for i in nearest_neighbors:
        cnt += 1
        # show_img(images1[i])
        # plt.imshow(images1[i])
        # plt.show()
        # img.show(images1[i])
        # print (j)
        # if (similarities[j]<0.8):
        # print (similarities[j])
        # print (j)
        j += 1
    # print (cnt)


def get_areaFromEquation(Xcoord, Ycoord, coordID, K, M):  # in Y > KX + M
    area = []
    _Xsolve = []
    _Ysolve = []
    j = 0
    for i in coordID:
        eq = 0
        eq = Xcoord[j] * K
        eq += M

        if Ycoord[j] > eq:
            area.append(coordID[j])
            _Xsolve.append(Xcoord[j])
            _Ysolve.append(Ycoord[j])
        j += 1
    return area, _Xsolve, _Ysolve


def get_areaFromLimit(Xarr, Yarr, CoordID, Xmin, Xmax, Ymin, Ymax):
    area = []
    _Xsolve = []
    _Ysolve = []
    j = 0
    for i in Xarr:

        if Xmin < Xarr[j] < Xmax and Ymin < Yarr[j] < Ymax:
            # print ("X",simil4[j],", Y", simil3[j],", #", neigh[j])
            area.append(CoordID[j])
            _Xsolve.append(Xarr[j])
            _Ysolve.append(Yarr[j])
        j += 1
    return area


# Send in the X and Y coords, the image array corresponding to the coordinates and specify icon size.
def plot_image_cluster(x_coord, y_coord, img_arr, icon_size, folder):
    import time
    images = []
    graph_size = (40, 40)

    fig, ax = plt.subplots(figsize=graph_size, dpi=100)
    ax.scatter(x_coord, y_coord)

    for filename in os.listdir(folder):

        if filename == "json.txt":
            continue

        _im = OffsetImage(plt.imread(os.path.join(folder, filename)), zoom=icon_size)
        images.append(_im)

    for x_coord, y_coord, Img in zip(x_coord, y_coord, images):
        ab = AnnotationBbox(Img, (x_coord, y_coord), frameon=False)
        ax.add_artist(ab)
    print("Plotting, time to sleep")
    plt.show()
    time.sleep(3600)


def run_example_INCEPTION_3(image_folder, model_path, create_new_model=True):

    current_model = Models.INCEPTION_3

    if create_new_model:
        base_model = create_model(model_path, current_model)
    else:
        base_model = load_model_from_path(model_path, current_model)

    print("Loading images..")
    images_arr = load_images_from_folder(image_folder, base_model)

    print("Getting predictions..")
    # Get predictions of images
    prediction_arr = get_predictions(images_arr, base_model, current_model)

    # Extract features from images
    # TODO: Is feature needed?
    # feature = get_features(images_arr, base_model)

    return prediction_arr
