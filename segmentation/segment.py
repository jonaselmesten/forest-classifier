from PIL import Image
from deepforest import main
from matplotlib import pyplot as plt


class BoundingBoxPredictor:

    def __init__(self):
        self.model = main.deepforest()
        self.model.use_release()

    def predict_trees(self, image_path, score_threshold=0, tree_count_threshold=None, show_prediction=False):
        """
        Will run a bounding box prediction on a given image.
        Returns a list of images and the bounding box data.

        @param image_path: Image to run prediction on.
        @param score_threshold: Prediction accuracy threshold.
        @param tree_count_threshold: Max tree count.
        @param show_prediction: PIL-image list, bounding box data list.
        @return:
        """
        try:

            pil_img = Image.open(image_path)
            bounding_boxes = self.model.predict_image(path=image_path, return_plot=False)

            if show_prediction:
                plt.imshow(bounding_boxes[:, :, ::-1])
                plt.show()

            tree_count = len(bounding_boxes)
            img_list = []
            coordinate_list = []

            if tree_count_threshold is not None:
                if tree_count < tree_count_threshold:
                    print("Trees found:", tree_count)
                    print("Tree count is lower than the threshold")
                    return
            elif tree_count == 0:
                print("No trees found")
                return 0

            tree_count = 0

            for _, row in bounding_boxes.iterrows():

                x_min = round(row["xmin"])
                y_min = round(row["ymin"])
                x_max = round(row["xmax"])
                y_max = round(row["ymax"])
                score = round(row["score"], 2)

                if score < score_threshold:
                    continue

                tree_img = pil_img.crop((x_min, y_min, x_max, y_max))
                img_list.append(tree_img)
                coordinate_list.append(tuple([x_min, y_min, x_max, y_max]))

                tree_count += 1

        except IOError as e:
            print(e)
            return None
        else:
            return img_list, coordinate_list
