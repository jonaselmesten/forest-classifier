from PIL import Image
from deepforest import main
from matplotlib import pyplot as plt


class TreePredictor:

    def __init__(self):
        self.model = main.deepforest()
        self.model.use_release()

    def predict_trees(self,
                      image_path,
                      score_threshold=0,
                      tree_count_threshold=None,
                      show_prediction=False) -> list:

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
            return
        else:
            return img_list, coordinate_list
