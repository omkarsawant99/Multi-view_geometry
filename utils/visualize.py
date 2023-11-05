import cv2
import numpy as np

def drawEpipolarLines(img1, img2, pts1, pts2, F):
    epipolar_img1 = img1.copy()
    epipolar_img2 = img2.copy()
    c1 = epipolar_img1.shape[1]
    c2 = epipolar_img2.shape[1]

    for i in range(len(pts2)):
        l1 = F.T @ np.append(pts2[i, :], [1], 0)
        x0, y0 = map(int, [0, -l1[2]/l1[1]])
        x1, y1 = map(int, [c1, -(l1[2]+l1[0]*c1)/l1[1]])
        epipolar_img1 = cv2.line(epipolar_img1, (x0, y0), (x1, y1), (255, 0, 0), 1)
        epipolar_img1 = cv2.circle(epipolar_img1, (int(pts1[i, 0]), int(pts1[i, 1])), 5, (0, 0, 255), -1)

    for i in range(len(pts1)):
        l2 = F @ np.append(pts1[i, :], [1], 0)
        x0, y0 = map(int, [0, -l2[2]/l2[1]])
        x1, y1 = map(int, [c2, -(l2[2]+l2[0]*c2)/l2[1]])
        epipolar_img2 = cv2.line(epipolar_img2, (x0, y0), (x1, y1), (255, 0, 0), 1)
        epipolar_img2 = cv2.circle(epipolar_img2, (int(pts2[i, 0]), int(pts2[i, 1])), 5, (0, 0, 255), -1)

    return epipolar_img1, epipolar_img2
