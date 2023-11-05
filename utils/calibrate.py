import numpy as np
import cv2

def calibrateCamera(images, chess_size, frame_size, show_images=False):
    objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if show_images:
                cv2.drawChessboardCorners(image, chess_size, corners, ret)
                cv2.imshow('Chessboard', image)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
    return ret, mtx, dist, rvecs, tvecs

