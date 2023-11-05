import numpy as np
import cv2


chess_size = (6, 8)
frame_size = (1080, 1920)


def get_calibration_matrix(images, show_images=0):
    world_points = []
    img_points = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare world co-ordinates
    wrldpts = np.zeros((chess_size[0]*chess_size[1], 3), np.float32)
    wrldpts[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

    for fname in images:
        img = cv2.imread(fname)
        assert img is not None, "Image not loaded properly, check the path."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)

        # If found, add object points, image points
        if ret == True:
            world_points.append(wrldpts)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

        if show_images:
            cv2.drawChessboardCorners(img, chess_size, corners2, ret)
            cv2.imshow("Test", img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, gray.shape[::-1], None, None)

    return mtx
