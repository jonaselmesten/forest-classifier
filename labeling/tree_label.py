# Read prediction_list.csv, return list
def readCSV2List():
  filePath = "/content/drive/My Drive/CNN_lager_dataset/prediction_list_1.csv"
  try:
    file = open(filePath,'r', encoding="gbk")
    context = file.read()
    list_result = context.split("\n")
    length = len(list_result)
    for i in range(length):
      list_result[i] = list_result[i].split(",")
    return list_result
  except Exception:
      print("Read failed.")  

import os
def num_input_birch():
   # get number of images in images_labeled_input/birch for numbering images. 
    birchpath = "/content/drive/My Drive/CNN_lager_dataset/dataset/birch"
    read_birchpath = os.listdir(birchpath) # get image_folder path
    #num_birch = len(read_birchpath)
    num_birch = 0
    for item in read_birchpath:
      if not item.startswith('.ipy'):
        num_birch += 1
    return num_birch
print(num_input_birch())

def num_input_spruce():
    # get number of images in images_labeled_input/spruce for numbering images. 
    sprucepath = "/content/drive/My Drive/CNN_lager_dataset/dataset/spruce"
    read_sprucepath = os.listdir(sprucepath) # get image_folder path
    #print(len(read_sprucepath)) #will include .ipynb_checkpints
    num_spruce = 0
    for item in read_sprucepath:
      if not item.startswith('.ipy'):
        num_spruce += 1
    return num_spruce
print(num_input_spruce())


# rename these unlabeled trees with a prediction name
list_result = readCSV2List()
new_list = list_result[1:-1] # [[prediction, accuracy, image]..],delete list's header and the last empty line

def rename(predict_list):
  # Rename all the trees in the list
  
  # initialize number of tree
  num_birch = num_input_birch()- 1
  num_spruce = num_input_spruce() - 1

  for i in predict_list:
    prediction_name = i[0]
    image_path = i[2]
    # Note that if the path has changed, the split method need to be changed.
    image_name = i[2].split('/')[7] # treeXXX.png
    image_folder = image_path.split('unlabeled-images/')[0] + 'unlabeled-images' # /content/drive/My Drive/CNN_lager_dataset/images_tobeclassified/unlabeled-images
    image_format = '.' + image_path.split('.')[1]  # .png

    # get a new name, different from names in dataset
    if prediction_name == 'birch':
      num_birch += 1
      new_name = prediction_name + '_' + str(num_birch) + image_format 
    elif prediction_name == 'spruce':
      num_spruce += 1
      new_name = prediction_name + '_' + str(num_spruce) + image_format

    # save the new name into the folder
    src = os.path.join(os.path.abspath(image_folder), image_name)
    dst = os.path.join(os.path.abspath(image_folder), new_name)
    os.rename(src, dst)
    print('converting %s to %s ...'%(src, dst))

rename(new_list)

# def pine_to_birch(name):
#   # change all the pines' names into birch
#   path = "/content/drive/My Drive/labeling/dataset/pine"
#   read_path = os.listdir(path)
#   num_tree = len(read_path)
#   i = 0
#   for item in read_path:
#     if item.endswith('png'):
#       suffix = item.split("_")[1]
#       new_name = name + '_' + suffix
#       src = os.path.join(os.path.abspath(path), item)
#       dst = os.path.join(os.path.abspath(path), new_name)
#       try:
#         os.rename(src, dst)
#         i = i + 1
#       except:
#         print("Somthing wrong.")
#     print('total %d to rename & converted %d jpgs' %(num_tree, i))

# pine_to_birch("birch")


# function: move images from src_path to dst_path
import os
import shutil
import traceback

def move_file(src_path, dst_path):
  print('from : ',src_path)
  print('to : ',dst_path)

  dir_folder = os.listdir(src_path)
  for file in dir_folder:
    try:
      f_src = os.path.join(src_path, file)
      if not os.path.exists(dst_path):
        os.mkdir(dst_path)
      f_dst = os.path.join(dst_path, file)
      shutil.move(f_src, f_dst)
    except Exception as e:
      print('move_file ERROR: ',e)
      traceback.print_exc()

# move_file('/content/drive/My Drive/labeling/images/1','/content/drive/My Drive/labeling/images_labeled_input/spruce')

