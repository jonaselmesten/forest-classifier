
def test_accuracy(dir_path, class_name, classifier):
    """
    Tests the accuracy of the model given a directory only containing one class.
    @param dir_path: Path.
    @param class_name: Name of the class.
    @param classifier: Model.
    """
    trees = [os.path.join(r, file) for r, d, f in os.walk(dir_path) for file in f]

    tree_count = len(trees)
    error_count = 0
    under_80_count = 0
    under_70_count = 0
    under_60_count = 0
    under_50_count = 0
    lowest_acc = 100.0

    print("Classifying ", len(trees), " trees...")
    print("Class:", class_name)

    start_time = timeit.default_timer()

    for tree in trees:
        predicted_class, accuracy = classifier.classify_single_tree(tree)

        if accuracy < lowest_acc:
            lowest_acc = accuracy

        if accuracy < 80.0:
            under_80_count += 1
        if accuracy < 70.0:
            under_70_count += 1
        if accuracy < 60.0:
            under_60_count += 1
        if accuracy < 50.0:
            under_50_count += 1

        if predicted_class.lower() != class_name.lower():
            error_count += 1

    elapsed = timeit.default_timer() - start_time
    print("Took {time} seconds to classify each tree.".format(time=elapsed / len(trees)))
    print("Took {time} seconds to classify all trees.".format(time=elapsed))
    print("--------------------------")
    print("Lowest accuracy:", lowest_acc)
    print("Trees under 80.0 accuracy:", under_80_count)
    print("Trees under 70.0 accuracy:", under_70_count)
    print("Trees under 60.0 accuracy:", under_60_count)
    print("Trees under 50.0 accuracy:", under_50_count)
    print(error_count, " out of ", tree_count, " was predicted wrong.")
    print("--------------------------")