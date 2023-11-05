import cv2
import cv2.aruco as aruco
import numpy as np


markerSize = 6 
totalMarkers=250


def detectArucoTag(img, markerSize, totalMarkers, dictionary_type=aruco.DICT_6X6_250):
    arucoDict = aruco.getPredefinedDictionary(dictionary_type)
    arucoParams = aruco.DetectorParameters_create()
    (corners, ids, rejected) = aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    return corners, ids


def prepare_corner_points(corners, ids):
    # Assuming you want the centers of the detected markers
    centers = []
    for corner in corners:
        center = corner[0].mean(axis=0)
        centers.append(center)
    return np.array(centers, dtype=np.float32), ids


def process_image_for_aruco(img, markerSize, totalMarkers):
    corners, ids = detectArucoTag(img, markerSize, totalMarkers)
    if ids is not None:
        pts, ids = prepare_corner_points(corners, ids)
        return pts, ids
    else:
        return None, None

