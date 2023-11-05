import cv2
import numpy as np


def calculate_fundamental_matrix(pts1, pts2):
    # Using OpenCV's function to compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    return F


def calculate_essential_matrix(F, K):
    E = K.T @ F @ K
    return E


def estimate_camera_pose(E, K, pts1, pts2):
    # This is a simplification and uses OpenCV's recoverPose function
    # Normally you would decompose E to get possible solutions and test them
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t
