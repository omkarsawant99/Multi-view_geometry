import cv2
import numpy as np

def draw_epipolar_lines(img1, img2, F, pts1, pts2):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines_left = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines_left = lines_left.reshape(-1, 3)
    img1_with_lines = img1.copy()
    for r, pt1 in zip(lines_left, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1] ])
        img1_with_lines = cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)
        img1_with_lines = cv2.circle(img1_with_lines, tuple(pt1), 5, color, -1)
    return img1_with_lines
