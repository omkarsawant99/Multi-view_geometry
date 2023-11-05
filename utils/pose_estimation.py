import numpy as np
import cv2

def normalizePoints(pts):
    mean = np.mean(pts, axis=0)
    max_dist = np.sqrt(((pts - mean) ** 2).sum(axis=1)).max()
    scale = np.sqrt(2) / max_dist

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])

    pts_homogeneous = np.column_stack((pts, np.ones(pts.shape[0])))
    pts_normalized = (T @ pts_homogeneous.T).T

    return pts_normalized, T


def getFundamentalMatrix(pts1, pts2):
    # Using 8-point algorithm with normalized coordinates for stability
    pts1_normalized, T1 = normalizePoints(pts1)
    pts2_normalized, T2 = normalizePoints(pts2)

    A = np.zeros((len(pts1), 9))
    for i, (p1, p2) in enumerate(zip(pts1_normalized, pts2_normalized)):
        A[i] = [p2[0] * p1[0], p2[0] * p1[1], p2[0], p2[1] * p1[0], p2[1] * p1[1], p2[1], p1[0], p1[1], 1]
    
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0
    F = Uf @ np.diag(Sf) @ Vtf

    # Denormalize the fundamental matrix
    F = T2.T @ F @ T1

    return F


def calculateEssentialMatrix(F, K):
    return K.T @ F @ K


def decomposeEssentialMatrix(E):
    # Assuming E is the essential matrix
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Two possible translations
    t1 = U[:, 2]
    t2 = -U[:, 2]

    return R1, R2, t1, t2
