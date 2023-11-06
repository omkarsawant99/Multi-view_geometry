# Camera Pose Estimation
This repository contains a collection of Python modules that work together to perform camera pose estimation using 2D images. The process involves detecting ArUco markers in stereo images, calculating the fundamental and essential matrices, and estimating the camera poses based on these calculations. This can be used in various applications such as augmented reality, 3D reconstruction, and robotics.

## Purpose
The code aims to estimate the relative pose (position and orientation) of a camera based on images of a known pattern (e.g., a chessboard for calibration) and ArUco markers for pose estimation. The following steps are performed:

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

## Notes
Matplotlib is used for visualization, so ensure you have a display environment configured if running this code in a headless setup.
