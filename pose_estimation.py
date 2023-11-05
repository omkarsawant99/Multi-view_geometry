import cv2
import numpy as np

def normalize_points(pts):
    pts_mean = np.mean(pts, axis=0)
    offset_pts = pts - pts_mean
    scale = np.sqrt(np.sum(offset_pts**2) / pts.size)

    T = np.array([
        [1/scale, 0,      -pts_mean[0]/scale],
        [0,       1/scale, -pts_mean[1]/scale],
        [0,       0,       1]
    ])

    norm_pts = np.column_stack((offset_pts / scale, np.ones(len(pts))))
    return norm_pts, T


def calculate_fundamental_matrix(pts1, pts2):
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    A = np.column_stack([norm_pts1[:, 0]*norm_pts2[:, 0], norm_pts1[:, 1]*norm_pts2[:, 0], norm_pts2[:, 0],
                         norm_pts1[:, 0]*norm_pts2[:, 1], norm_pts1[:, 1]*norm_pts2[:, 1], norm_pts2[:, 1],
                         norm_pts1[:, 0], norm_pts1[:, 1], np.ones(len(norm_pts1))])

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 by zeroing the smallest singular value
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0

    F_norm = U @ np.diag(S) @ Vt

    F = T2.T @ F_norm @ T1

    return F

'''
def calculate_fundamental_matrix(pts1, pts2):
    # Using OpenCV's function to compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    return F
'''

def calculate_essential_matrix(F, K):
    E = K.T @ F @ K
    return E


def estimate_camera_pose(E, K, pts1, pts2):
    # This is a simplification and uses OpenCV's recoverPose function
    # Normally you would decompose E to get possible solutions and test them
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t
