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


def calculate_essential_matrix(F, K):
    E = K.T @ F @ K
    return E


def estimate_camera_poses(E, K, pts1, pts2):
    Y = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    U, S, Vt = np.linalg.svd(E)

    # Ensure that U and Vt are rotations
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    # Create two possible rotations
    R1 = U @ Y @ Vt
    R2 = U @ Y.T @ Vt

    # Translation is the third column of U
    t = U[:, 2].reshape(-1, 1)

    return [R1, R2, R1, R2], [t, t, -t, -t]


def triangulate_point(p1, p2, T1, T2):
    """
    Triangulate a single point from two projections
    INPUT:  p1, p2 = points in homogeneous coordinates in image 1 and 2
            T1, T2 = projection matrices for the two images (K * [R | t])
    
    OUTPUT: X = 3D co-ordinates of corresponding image point (p1, p2) in reference frame (first camera pose)
    """
    # Create the matrix A for the homogeneous equation system Ax = 0
    A = np.zeros((4, 4))
    A[0, :] = p1[0] * T1[2, :] - T1[0, :]
    A[1, :] = p1[1] * T1[2, :] - T1[1, :]
    A[2, :] = p2[0] * T2[2, :] - T2[0, :]
    A[3, :] = p2[1] * T2[2, :] - T2[1, :]
    
    # Solve the equation Ax = 0 by SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[-1]

    return X[:-1]


def chirality_check(Rs, ts, K, pts1, pts2):
    """
    Perform chirality check to determine which configuration is valid
    INPUT:  Rs, ts = lists of rotation and translation matrices
            K = intrinsic camera matrix
            pts1, pts2 = matching points in each image
    
    OUPUT
    """
    # Identity matrix for the first camera pose
    T1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    # Number of points
    num_points = pts1.shape[0]

    # Normalize points
    norm_pts1, _ = normalize_points(pts1)
    norm_pts2, _ = normalize_points(pts2)

    # The correct configuration is the one with the most points in front of both cameras
    max_positive_depth = -np.inf
    correct_configuration = None

    for R, t in zip(Rs, ts):
        T2 = np.hstack((R, t))

        # Project to camera coordinates based on pinhole camera model
        # [u, v, 1] = (K * [R | t]) * [X, Y, Z, 1]
        T1 = K @ T1
        T2 = K @ T2

        # Count how many points are in front of both cameras
        positive_depth_count = 0
        for i in range(num_points):
            X = triangulate_point(norm_pts1[i], norm_pts2[i], T1, T2)

            # Check if the point is in front of both cameras
            if (R[2, :] @ (X[:3] - t.squeeze())) > 0 and X[2] > 0:
                positive_depth_count += 1

        # Update the best configuration if needed
        if positive_depth_count > max_positive_depth:
            max_positive_depth = positive_depth_count
            correct_configuration = (R, t)

    return correct_configuration

