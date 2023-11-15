# Camera Pose Estimation
This repository contains a collection of Python modules that work together to perform camera pose estimation using 2D images. The process involves detecting corners of ArUco markers in stereo images, calculating the fundamental and essential matrices, and estimating the camera poses based on these calculations.
The ArUco markers are used only for accurate feature detection and matching, without outliers. 

## Purpose
The code aims to estimate the relative pose (position and orientation) of a camera based on images. The following steps are performed:

* **Camera Calibration**: Determine the camera's intrinsic parameters to correct lens distortion.

* **Fundamental Matrix Calculation**: Compute the fundamental matrix using corresponding points from stereo images.

* **Essential Matrix Calculation**: Derive the essential matrix from the fundamental matrix using the camera's intrinsic parameters.

* **Camera Pose Estimation**: Estimate the rotation and translation of the camera with respect to a world coordinate system.

* **Chirality Check**: Determine the correct camera pose among possible solutions by ensuring that points are in front of the camera.

## Dependencies
To run this code, you will need the following libraries:
numpy
opencv-python
matplotlib
You can install these packages using pip:

`pip install numpy opencv-python matplotlib`

## How to Run the Code
Run the main.py script to start the pose estimation process.

`python main.py`

## Visualization
Matplotlib is used for visualization, so ensure you have a display environment configured if running this code in a headless setup.
The visualization should look like this -

<img width="434" alt="image" src="https://github.com/omkarsawant99/Multi-view_geometry/assets/112906388/1a4baab0-6002-4949-a48b-d7af4c7e77c5">
