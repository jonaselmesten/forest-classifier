# 3D-forest

![plot](./resources/trees.JPG)
<br>

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Workflow](#workflow)

## Attention!
All images in this file does not represent a perfect result achieved by this program! The flickering (int the gif) of some trees and
the bad accuracy is due to a low-res, low-accuracy point cloud used for testing. The code is by no means restricted 
to vertically segmented objects in point clouds. Could just as well be used for point clouds created by drones from above.

## General info
This program automatically creates annotation data from segmented objects in SfM point clouds.
It aims to reduce the need for manual annotation of images when preparing them for neural networks like
MASK R-CNN, where segmented & labeled training data is needed.
It works by creating 2D-representations of segmented 3D objects (trees in this case) and bundles all pixel masks
to a single instance of PointObject. The instance of this object can then be manipulated in terms of the 2D-area it covers, 
where updates are propagated through the image sequence. Thus reducing the need for manual corrections if the original masks is not of 
satisfactory accuracy.

![plot](./resources/colmap.png)
SfM point cloud created with COLMAP.

## Technologies
To use this project:
* COLMAP SfM point cloud generator https://colmap.github.io/index.html
* TreeSeg Point cloud tree segmentation package https://github.com/apburt/treeseg 
  (or other object segmentation algorithm, see below)
* Python 3.8

## Workflow
* Create a SfM point cloud with COLMAP. 
    * Put the used images in input/images.
    * Put the images.txt and points3D.txt in input/model_input.
  
* Use TreeSeg to segment the objects to be used for pixel mask creation.
  <br> This is done separately, any object segmentation algorithm can be used for segmentation.
  All segmented objects should be saved as a separate PCD file of the following format 
  (the original POINT3D_ID must be preserved through the segmentation):
  ```
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z id
    SIZE 4 4 4 4
    TYPE F F F I
    COUNT 1 1 1 1
    WIDTH 2014
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 2014
    DATA ascii
    -0.35527748 -1.2384315 7.9378061 390693
    -0.33041301 -0.81031048 7.9623089 390686
    -0.2881591 -0.9084934 7.9559093 389515
    ...........
  ```
    * Place all object.pcd files in input/pcd_input.
* Run the program!

Take a look at visualize/interface.py for examples of what can be done with all the point data structures!
In this file you'll find everything that's needed to understand the flow of execution of this program. 
If you want to modify something/get a feel for everything, I suggest you start here and look at everything that's loaded
in the top of the file and at the example-functions.

![plot](./resources/flow.gif)
<br>
Automatically annotated sequence of pixel masks. The label for each separate tree can be set easily.
The result can then be saved in CSV. See files/csv_file.py.



## Setup
```
```

