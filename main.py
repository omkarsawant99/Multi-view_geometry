import cv2
import numpy as np
from detector import process_image_for_aruco
from pose_estimation import calculate_fundamental_matrix, calculate_essential_matrix, estimate_camera_pose
from epipolar import draw_epipolar_lines
from calibration import get_calibration_matrix
from visualizer import visualize_camera_poses
import os
import glob


# Import stereo images
left = cv2.imread('content/left.jpg')
right = cv2.imread('content/right.jpg')
assert left is not None, "Image not loaded properly, check the path."
assert right is not None, "Image not loaded properly, check the path."

# Import calibration images 
images = glob.glob('calib_images/*.jpg')


if __name__ == "__main__":
    markerSize = 6
    totalMarkers = 250
    K = get_calibration_matrix(images)
    
    # Ensure that 'left.jpg' and 'right.jpg' are the paths to your images
    pts1, ids1 = process_image_for_aruco(left, markerSize, totalMarkers)
    pts2, ids2 = process_image_for_aruco(right, markerSize, totalMarkers)

    if pts1 is not None and pts2 is not None and K is not None:
        F = calculate_fundamental_matrix(pts1, pts2)
        E = calculate_essential_matrix(F, K)
        R, t = estimate_camera_pose(E, K, pts1, pts2)
        print("Estimated Camera Pose:")
        print("Rotation Matrix:\n", R)
        print("Translation Vector:\n", t)

        R0 = np.eye(3)
        t0 = np.zeros(3)
        visualize_camera_poses(R0, t0, R, t)

        img1_with_lines = draw_epipolar_lines(left, right, F, pts1, pts2)
        # You may want to save or display the image using cv2.imshow and cv2.imwrite

    else:
        print("Aruco markers not detected in both images or missing camera intrinsic matrix")


